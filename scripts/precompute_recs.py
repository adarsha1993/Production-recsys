import sys
import json
import torch
import joblib
import pandas as pd
from pathlib import Path

sys.path.insert(0, '.')

CKPT = Path('models/checkpoints')
PROC = Path('data/processed')

print("Loading data...")
vocab = joblib.load(
    CKPT / 'item_vocabulary.joblib')
user_seqs = joblib.load(
    CKPT / 'user_sequences.joblib')
token2movie = {
    int(k): v
    for k, v in
    vocab['token2movie'].items()}

movies = pd.read_csv(
    PROC / 'movies_master.csv',
    low_memory=False)
movies['movieId'] = pd.to_numeric(
    movies['movieId'], errors='coerce')
movies = movies.dropna(
    subset=['movieId'])
movies['movieId'] = \
    movies['movieId'].astype(int)

ratings = pd.read_csv(
    PROC / 'ratings_cleaned.csv')

title_map  = dict(zip(
    movies['movieId'], movies['title']))
poster_map = dict(zip(
    movies['movieId'],
    movies['poster_path']))
pop = ratings.groupby(
    'movieId')['rating']\
    .count()\
    .sort_values(ascending=False)\
    .index.tolist()

print("Loading HSTU model...")
from src.serving.bentoml_service import (
    model, PAD_TOKEN)

all_recs = {}
users    = list(user_seqs.keys())
print(f"Computing recs for {len(users)} users...")

for i, uid in enumerate(users):
    seq = user_seqs.get(uid, [])
    try:
        rated = set(
            int(m) for m in
            ratings[
                ratings['userId'] == uid
            ]['movieId'].values)
    except Exception:
        rated = set()

    seen = set()
    recs = []

    if seq:
        pad_l = 50 - len(seq)
        hist  = torch.LongTensor(
            [[PAD_TOKEN]*pad_l + seq[-50:]])
        with torch.no_grad():
            indices, values = model.predict(
                hist, top_k=200)
        sc_arr   = values[0].cpu().numpy()
        sc_min   = sc_arr.min()
        sc_range = max(
            sc_arr.max()-sc_min, 1e-6)

        for tok, sc in zip(
                indices[0].cpu().numpy(),
                sc_arr):
            mid = token2movie.get(int(tok))
            if not mid:
                continue
            mid = int(mid)
            if mid in rated or \
                    mid in seen:
                continue
            seen.add(mid)
            poster = str(
                poster_map.get(mid, ''))
            recs.append({
                'rank':       len(recs)+1,
                'movie_id':   mid,
                'title':      str(
                    title_map.get(
                        mid,
                        f'Movie {mid}')),
                'score':      round(float(
                    (sc-sc_min)/sc_range),
                    4),
                'poster':     poster
                              if poster
                              not in (
                                'nan',
                                '',
                                'None')
                              else '',
                'fallback':   False,
                'cold_start': False,
            })
            if len(recs) >= 20:
                break

    for mid in pop:
        if len(recs) >= 20:
            break
        mid = int(mid)
        if mid in rated or mid in seen:
            continue
        seen.add(mid)
        poster = str(
            poster_map.get(mid, ''))
        recs.append({
            'rank':       len(recs)+1,
            'movie_id':   mid,
            'title':      str(
                title_map.get(
                    mid,
                    f'Movie {mid}')),
            'score':      0.001,
            'poster':     poster
                          if poster
                          not in (
                            'nan',
                            '',
                            'None')
                          else '',
            'fallback':   True,
            'cold_start': False,
        })

    for j, r in enumerate(recs):
        r['rank'] = j + 1

    all_recs[str(uid)] = recs

    if (i+1) % 100 == 0:
        print(f"  {i+1}/{len(users)} done")

out = PROC / 'precomputed_recs.json'
with open(out, 'w') as f:
    json.dump(all_recs, f)

size = out.stat().st_size / 1e6
print(f"\nDone!")
print(f"  Users : {len(all_recs)}")
print(f"  Size  : {size:.1f}MB")
print(f"  File  : {out}")
