"""
BentoML Production Service (1.x API)
Using Pydantic models for proper input parsing.

Usage:
  bentoml serve src.serving.bentoml_service:HSTURankerService --port 3001
"""

import time
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import bentoml
from pydantic import BaseModel
from typing import Optional, List
from pathlib import Path

# ── Paths ─────────────────────────────────────────
BASE = Path(__file__).parent.parent.parent
CKPT = BASE / 'models' / 'checkpoints'
PROC = BASE / 'data' / 'processed'


# ── Pydantic Models ───────────────────────────────
class RecommendRequest(BaseModel):
    user_id: int = 0
    top_k:   int = 10

class FeedbackRequest(BaseModel):
    user_id:  int   = 0
    movie_id: int   = 0
    rating:   float = 0.0
    action:   str   = "watch"

class HealthRequest(BaseModel):
    ping: str = "ping"


# ── HSTU Architecture ─────────────────────────────
class HSTULayer(nn.Module):
    def __init__(self, embed_dim,
                 n_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads   = n_heads
        self.head_dim  = embed_dim // n_heads
        self.q_proj    = nn.Linear(
            embed_dim, embed_dim, bias=False)
        self.k_proj    = nn.Linear(
            embed_dim, embed_dim, bias=False)
        self.v_proj    = nn.Linear(
            embed_dim, embed_dim, bias=False)
        self.out_proj  = nn.Linear(
            embed_dim, embed_dim, bias=False)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.scale   = self.head_dim ** -0.5

    def forward(self, x, src_mask=None):
        B, L, D  = x.shape
        residual = x
        x        = self.norm1(x)
        q = self.q_proj(x).view(
            B, L, self.n_heads,
            self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(
            B, L, self.n_heads,
            self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(
            B, L, self.n_heads,
            self.head_dim).transpose(1, 2)
        attn = torch.matmul(
            q, k.transpose(-2, -1)
        ) * self.scale
        attn = F.relu(attn)
        s    = attn.sum(
            dim=-1, keepdim=True
        ).clamp(min=1e-6)
        attn = attn / s
        attn = self.dropout(attn)
        out  = torch.matmul(attn, v)
        out  = out.transpose(1, 2)\
            .contiguous().view(B, L, D)
        x    = residual + self.out_proj(out)
        x    = x + self.ff(self.norm2(x))
        return x


class HSTURanker(nn.Module):
    def __init__(self,
                 vocab_size,
                 embed_dim=128,
                 n_heads=4,
                 n_layers=3,
                 max_seq_len=50,
                 dropout=0.1,
                 pad_token=0,
                 offset=3):
        super().__init__()
        self.pad_token = pad_token
        self.offset    = offset
        self.item_emb  = nn.Embedding(
            vocab_size, embed_dim,
            padding_idx=pad_token)
        self.pos_emb   = nn.Embedding(
            max_seq_len, embed_dim)
        self.hstu_layers = nn.ModuleList([
            HSTULayer(embed_dim,
                      n_heads, dropout)
            for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(
            embed_dim)
        self.dropout    = nn.Dropout(dropout)
        self.rating_head = nn.Sequential(
            nn.Linear(embed_dim,
                      embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid())
        self.completion_head = nn.Sequential(
            nn.Linear(embed_dim,
                      embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid())

    def encode_user(self, history):
        B, L = history.shape
        pos  = torch.arange(
            L, device=history.device
        ).unsqueeze(0).expand(B, -1)
        x    = self.item_emb(history) + \
               self.pos_emb(pos)
        x    = self.dropout(x)
        for layer in self.hstu_layers:
            x = layer(x)
        x       = self.final_norm(x)
        lengths = (
            history != self.pad_token
        ).sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        return x[torch.arange(B), lengths]

    def predict(self, history, top_k=10):
        with torch.no_grad():
            u  = self.encode_user(history)
            sc = torch.matmul(
                u, self.item_emb.weight.T)
            sc[:, :self.offset] = \
                float('-inf')
            return torch.topk(
                sc, top_k, dim=-1)


# ── Load model + vocab once at startup ────────────
def load_model():
    vocab     = joblib.load(
        CKPT / 'item_vocabulary.joblib')
    user_seqs = joblib.load(
        CKPT / 'user_sequences.joblib')

    PAD_TOKEN   = vocab['PAD']
    vocab_size  = vocab['vocab_size']
    movie2token = vocab['movie2token']
    token2movie = {
        int(k): v
        for k, v in
        vocab['token2movie'].items()
    }

    _model = HSTURanker(
        vocab_size = vocab_size,
        pad_token  = PAD_TOKEN,
        offset     = 3,
    )
    _model.load_state_dict(torch.load(
        CKPT / 'hstu_best.pt',
        map_location='cpu',
        weights_only=True))
    _model.eval()

    return (_model, vocab, user_seqs,
            movie2token, token2movie,
            PAD_TOKEN)


# Load at module level
(model, vocab, user_seqs,
 movie2token, token2movie,
 PAD_TOKEN) = load_model()

# Load titles
try:
    import pandas as pd
    movies_df = pd.read_csv(
        PROC / 'movies_master.csv',
        low_memory=False)
    movies_df = movies_df[
        movies_df['movieId'].notna()
    ].copy()
    movies_df['movieId'] = \
        movies_df['movieId'].astype(int)
    title_map = dict(zip(
        movies_df['movieId'],
        movies_df['title']))
except Exception:
    title_map = {}

print("✅ HSTU model loaded for serving")
print(f"   Vocab size : {vocab['vocab_size']}")
print(f"   Users      : {len(user_seqs)}")
print(f"   Titles     : {len(title_map)}")


# ── BentoML 1.x Service ───────────────────────────
@bentoml.service(
    name="hstu_ranker",
    resources={"cpu": "2"},
    traffic={"timeout": 60},
)
class HSTURankerService:
    """
    BentoML 1.x production service.
    Uses Pydantic models for correct
    input parsing in BentoML 1.x.
    """

    def __init__(self):
        self.model         = model
        self.user_seqs     = user_seqs
        self.movie2token   = movie2token
        self.token2movie   = token2movie
        self.title_map     = title_map
        self.PAD_TOKEN     = PAD_TOKEN
        self.request_count = 0

    def _get_recs(self,
                  user_id: int,
                  top_k:   int = 10
                  ) -> list:
        seq = self.user_seqs.get(
            user_id, [])

        print(f"Getting recs for user "
              f"{user_id} — "
              f"seq_len={len(seq)}")

        if not seq:
            return [
                {
                    "movie_id":   -1,
                    "title":      "Popular Movie",
                    "score":      0.0,
                    "rank":       i + 1,
                    "cold_start": True,
                }
                for i in range(top_k)
            ]

        pad_l = 50 - len(seq)
        hist  = torch.LongTensor(
            [[self.PAD_TOKEN] * pad_l +
              seq[-50:]])

        toks, scores = self.model.predict(
            hist, top_k=min(top_k * 10, 500))

        rated    = set(seq)
        seen_mid = set()
        recs     = []

        sc_arr   = scores[0].cpu().numpy()
        sc_min   = sc_arr.min()
        sc_max   = sc_arr.max()
        sc_range = max(sc_max - sc_min, 1e-6)

        for tok, sc in zip(
                toks[0].cpu().numpy(),
                sc_arr):
            mid = self.token2movie.get(
                int(tok))
            if not mid:
                continue
            if int(tok) in rated:
                continue
            if mid in seen_mid:
                continue

            seen_mid.add(mid)
            title      = self.title_map.get(
                mid, f"Movie {mid}")
            norm_score = float(
                (sc - sc_min) / sc_range)

            recs.append({
                "movie_id":   int(mid),
                "title":      str(title),
                "score":      round(
                    norm_score, 4),
                "rank":       len(recs)+1,
                "cold_start": False,
                "fallback":   False,
            })

            if len(recs) >= top_k:
                break

        # ── Popularity padding ────────────────
        # If HSTU found fewer than top_k
        # pad with popular unrated movies
        if len(recs) < top_k:
            try:
                import pandas as pd
                ratings_df = pd.read_csv(
                    PROC / 'ratings_cleaned.csv')
                pop = ratings_df.groupby(
                    'movieId')['rating']\
                    .count()\
                    .sort_values(
                        ascending=False)

                rated_mids = set(
                    self.token2movie.get(
                        int(t), -1)
                    for t in seq)

                for mid in pop.index:
                    if len(recs) >= top_k:
                        break
                    mid = int(mid)
                    if mid not in rated_mids \
                       and mid not in seen_mid:
                        title = self.title_map\
                            .get(mid,
                                 f"Movie {mid}")
                        seen_mid.add(mid)
                        recs.append({
                            "movie_id":   mid,
                            "title":      str(title),
                            "score":      0.001,
                            "rank":       len(recs)+1,
                            "cold_start": False,
                            "fallback":   True,
                        })
            except Exception as e:
                print(f"Padding failed: {e}")

        # Re-number ranks after padding
        for i, r in enumerate(recs):
            r['rank'] = i + 1

        return recs
    @bentoml.api()
    def recommend(
            self,
            request: RecommendRequest
    ) -> dict:
        """
        POST /recommend
        Body: {"user_id": 123, "top_k": 10}
        """
        start   = time.time()

        print(f"recommend called: "
              f"user_id={request.user_id} "
              f"top_k={request.top_k}")

        self.request_count += 1
        recs    = self._get_recs(
            request.user_id,
            request.top_k)
        latency = (time.time()-start)*1000

        return {
            "user_id":         request.user_id,
            "recommendations": recs,
            "model":           "HSTU",
            "latency_ms":      round(
                latency, 2),
            "n_recs":          len(recs),
            "request_count":   self.request_count,
        }

    @bentoml.api()
    def health(
            self,
            request: HealthRequest
    ) -> dict:
        """POST /health — liveness probe"""
        return {
            "status":        "healthy",
            "model":         "HSTU",
            "vocab_size":    vocab[
                'vocab_size'],
            "n_users":       len(user_seqs),
            "n_titles":      len(title_map),
            "request_count": self.request_count,
            "timestamp":     time.time(),
        }

    @bentoml.api()
    def feedback(
            self,
            request: FeedbackRequest
    ) -> dict:
        """
        POST /feedback
        Body: {
          "user_id": 123,
          "movie_id": 456,
          "rating": 4.5,
          "action": "watch"
        }
        """
        print(f"Feedback: user="
              f"{request.user_id} "
              f"movie={request.movie_id} "
              f"rating={request.rating}")

        return {
            "status":    "logged",
            "user_id":   request.user_id,
            "movie_id":  request.movie_id,
            "rating":    request.rating,
            "action":    request.action,
            "timestamp": time.time(),
            "message":   "Feedback queued "
                         "for Kafka pipeline",
        }


# Alias
svc = HSTURankerService