"""
Production RecSys — Streamlit Frontend
Cinematic dark UI with film-grain aesthetic.

Usage:
  streamlit run src/frontend/streamlit_app.py
"""

import sys
import ast
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import streamlit as st

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))

# ── Config ────────────────────────────────────────
API_URL   = "http://localhost:8000"
PROM_URL  = "http://localhost:9090"
TMDB_BASE = "https://image.tmdb.org/t/p/w300"

st.set_page_config(
    page_title = "CineRec · AI Recommendations",
    page_icon  = "🎬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Design System ─────────────────────────────────
# Palette: deep cinema blacks + amber gold accent
# Inspired by 35mm film and projection light
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root tokens ── */
:root {
  --black:      #080808;
  --deep:       #0f0f0f;
  --surface:    #161616;
  --elevated:   #1f1f1f;
  --border:     #2a2a2a;
  --amber:      #f5a623;
  --amber-dim:  #c47f10;
  --amber-glow: rgba(245,166,35,0.12);
  --text-hi:    #f0ede8;
  --text-mid:   #a09890;
  --text-lo:    #555550;
  --green:      #4ade80;
  --red:        #f87171;
  --blue:       #60a5fa;
}

/* ── Global reset ── */
.stApp {
  background: var(--black) !important;
  font-family: 'Inter', sans-serif;
}

/* Film grain overlay */
.stApp::before {
  content: '';
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 9999;
  opacity: 0.4;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--deep) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
  color: var(--text-mid) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 12px !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
}

/* ── Headings ── */
h1, h2, h3 {
  font-family: 'Bebas Neue', sans-serif !important;
  letter-spacing: 0.08em !important;
  color: var(--text-hi) !important;
}

/* ── Hero title ── */
.hero-title {
  font-family: 'Bebas Neue', sans-serif;
  font-size: clamp(48px, 8vw, 96px);
  letter-spacing: 0.12em;
  color: var(--text-hi);
  line-height: 0.9;
  margin-bottom: 4px;
}
.hero-sub {
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--amber);
  letter-spacing: 0.25em;
  text-transform: uppercase;
  margin-bottom: 24px;
}

/* ── Amber accent line ── */
.amber-line {
  width: 60px;
  height: 3px;
  background: linear-gradient(
    90deg, var(--amber), transparent);
  margin: 12px 0 20px;
}

/* ── Movie card ── */
.movie-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0;
  overflow: hidden;
  transition: all 0.25s ease;
  position: relative;
  cursor: pointer;
}
.movie-card:hover {
  border-color: var(--amber-dim);
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(0,0,0,0.6),
              0 0 0 1px var(--amber-dim);
}
.movie-poster-placeholder {
  width: 100%;
  aspect-ratio: 2/3;
  background: linear-gradient(
    160deg, #1a1a1a 0%, #0d0d0d 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 48px;
  position: relative;
}
.movie-poster-placeholder::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(
    to bottom,
    transparent 60%,
    rgba(0,0,0,0.8) 100%);
}
.rank-badge {
  position: absolute;
  top: 8px;
  left: 8px;
  background: var(--amber);
  color: var(--black);
  font-family: 'Bebas Neue', sans-serif;
  font-size: 14px;
  letter-spacing: 0.08em;
  padding: 2px 8px;
  border-radius: 3px;
  z-index: 10;
}
.hstu-badge {
  position: absolute;
  top: 8px;
  right: 8px;
  background: rgba(15,15,15,0.9);
  border: 1px solid var(--amber-dim);
  color: var(--amber);
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.1em;
  padding: 2px 6px;
  border-radius: 3px;
  z-index: 10;
}
.pop-badge {
  position: absolute;
  top: 8px;
  right: 8px;
  background: rgba(15,15,15,0.9);
  border: 1px solid #444;
  color: var(--text-mid);
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px;
  letter-spacing: 0.1em;
  padding: 2px 6px;
  border-radius: 3px;
  z-index: 10;
}
.movie-info {
  padding: 10px 10px 12px;
  background: var(--surface);
}
.movie-title-card {
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  font-size: 12px;
  color: var(--text-hi);
  line-height: 1.3;
  margin-bottom: 6px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.score-bar-wrap {
  background: var(--elevated);
  border-radius: 2px;
  height: 3px;
  overflow: hidden;
  margin-bottom: 4px;
}
.score-bar-fill {
  height: 100%;
  background: linear-gradient(
    90deg, var(--amber-dim), var(--amber));
  border-radius: 2px;
  transition: width 0.6s ease;
}
.score-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text-lo);
  letter-spacing: 0.08em;
}

/* ── Stat cards ── */
.stat-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  text-align: center;
  transition: border-color 0.2s;
}
.stat-card:hover {
  border-color: var(--amber-dim);
}
.stat-value {
  font-family: 'Bebas Neue', sans-serif;
  font-size: 36px;
  letter-spacing: 0.06em;
  color: var(--amber);
  line-height: 1;
}
.stat-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text-lo);
  letter-spacing: 0.2em;
  text-transform: uppercase;
  margin-top: 4px;
}

/* ── Status pill ── */
.pill-ok {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(74,222,128,0.08);
  border: 1px solid rgba(74,222,128,0.3);
  color: var(--green);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  padding: 4px 12px;
  border-radius: 20px;
}
.pill-warn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(248,113,113,0.08);
  border: 1px solid rgba(248,113,113,0.3);
  color: var(--red);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  padding: 4px 12px;
  border-radius: 20px;
}
.pill-cached {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--amber-glow);
  border: 1px solid var(--amber-dim);
  color: var(--amber);
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  letter-spacing: 0.1em;
  padding: 4px 12px;
  border-radius: 20px;
}

/* ── Section label ── */
.section-eyebrow {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--amber);
  letter-spacing: 0.3em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

/* ── History row ── */
.history-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  transition: background 0.15s;
}
.history-row:hover {
  background: var(--surface);
}
.history-title {
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-hi);
}
.history-rating {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--amber);
}

/* ── Architecture box ── */
.arch-box {
  background: var(--surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--amber);
  border-radius: 0 8px 8px 0;
  padding: 20px 24px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: var(--text-mid);
  line-height: 1.8;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-lo) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  padding: 10px 20px !important;
  border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
  color: var(--amber) !important;
  border-bottom-color: var(--amber) !important;
}

/* ── Inputs ── */
.stSelectbox > div > div {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-hi) !important;
  border-radius: 6px !important;
}
.stSlider [data-baseweb="slider"] div {
  background: var(--amber) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: transparent !important;
  border: 1px solid var(--amber-dim) !important;
  color: var(--amber) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.15em !important;
  text-transform: uppercase !important;
  padding: 8px 20px !important;
  border-radius: 4px !important;
  transition: all 0.2s !important;
}
.stButton > button:hover {
  background: var(--amber-glow) !important;
  border-color: var(--amber) !important;
}

/* ── Spinner ── */
.stSpinner > div {
  border-top-color: var(--amber) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  color: var(--text-mid) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 11px !important;
  letter-spacing: 0.1em !important;
}

/* ── Divider ── */
hr {
  border-color: var(--border) !important;
  margin: 20px 0 !important;
}

/* ── General text ── */
p, span, label, div {
  color: var(--text-mid);
}

/* ── Scroll bar ── */
::-webkit-scrollbar {
  width: 4px;
}
::-webkit-scrollbar-track {
  background: var(--black);
}
::-webkit-scrollbar-thumb {
  background: var(--border);
  border-radius: 2px;
}
::-webkit-scrollbar-thumb:hover {
  background: var(--amber-dim);
}

/* ── Page load animation ── */
@keyframes fadeUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.fade-up {
  animation: fadeUp 0.5s ease forwards;
}

/* ── Pulse dot ── */
@keyframes pulse {
  0%, 100% { opacity: 1; }
  50%       { opacity: 0.3; }
}
.pulse-dot {
  display: inline-block;
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--green);
  animation: pulse 2s infinite;
  margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)


# ── Data Loaders ──────────────────────────────────
@st.cache_data(ttl=300)
def load_movies():
    try:
        df = pd.read_csv(
            BASE / 'data' / 'processed' /
            'movies_master.csv',
            low_memory=False)
        df = df[df['movieId'].notna()].copy()
        df['movieId'] = \
            df['movieId'].astype(int)
        df['title']   = df['title'].fillna(
            df['movieId'].astype(str).apply(
                lambda x: f"Movie {x}"))
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_ratings():
    try:
        df = pd.read_csv(
            BASE / 'data' / 'processed' /
            'ratings_cleaned.csv')
        return df.sort_values(
            ['userId', 'timestamp'])
    except Exception:
        return pd.DataFrame()


# ── API Helpers ───────────────────────────────────
def get_recommendations(
        user_id: int,
        top_k:   int = 10) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/recommend",
            json    = {"user_id": user_id,
                       "top_k":   top_k},
            timeout = 30)
        return r.json() \
            if r.status_code == 200 \
            else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def send_feedback(
        user_id:  int,
        movie_id: int,
        rating:   float,
        action:   str = "rate") -> bool:
    try:
        r = requests.post(
            f"{API_URL}/feedback",
            json    = {
                "user_id":  user_id,
                "movie_id": movie_id,
                "rating":   rating,
                "action":   action,
            },
            timeout = 10)
        return r.status_code == 200
    except Exception:
        return False


def get_health() -> dict:
    try:
        r = requests.get(
            f"{API_URL}/health", timeout=3)
        return r.json() \
            if r.status_code == 200 else {}
    except Exception:
        return {}


def get_metrics() -> dict:
    try:
        r = requests.get(
            f"{API_URL}/metrics", timeout=3)
        return r.json() \
            if r.status_code == 200 else {}
    except Exception:
        return {}


def get_cache_stats() -> dict:
    try:
        r = requests.get(
            f"{API_URL}/cache/stats",
            timeout=3)
        return r.json() \
            if r.status_code == 200 else {}
    except Exception:
        return {}


def get_poster_url(
        movie_row: pd.Series
        ) -> Optional[str]:
    try:
        poster = movie_row.get(
            'poster_path', '')
        if poster and \
                str(poster) not in (
                    'nan', '', 'None'):
            return f"{TMDB_BASE}{poster}"
    except Exception:
        pass
    return None


def safe_genres(val) -> list:
    """Parse genres safely"""
    if val is None:
        return []
    if isinstance(val, float):
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            result = ast.literal_eval(val)
            return result \
                if isinstance(result, list) \
                else []
        except Exception:
            return []
    return []


# ── Load Data ─────────────────────────────────────
movies  = load_movies()
ratings = load_ratings()

if not ratings.empty:
    user_counts     = ratings.groupby(
        'userId')['rating'].count()
    available_users = user_counts\
        .sort_values(ascending=False)\
        .index.tolist()
else:
    available_users = list(range(1, 51))


# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:8px 0 20px">
  <div style="font-family:'Bebas Neue',sans-serif;
              font-size:28px;
              letter-spacing:0.15em;
              color:#f0ede8;">
    CINEREC
  </div>
  <div style="font-family:'JetBrains Mono',monospace;
              font-size:9px;
              color:#f5a623;
              letter-spacing:0.3em;
              text-transform:uppercase;
              margin-top:2px;">
    Neural Recommendation Engine
  </div>
  <div style="width:40px;height:2px;
              background:linear-gradient(90deg,#f5a623,transparent);
              margin-top:10px;"></div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="section-eyebrow">Select User</div>
""", unsafe_allow_html=True)

    user_id = st.selectbox(
        "User",
        options = available_users[:100],
        index   = 0,
        label_visibility = "collapsed")

    top_k = st.slider(
        "Recommendations",
        min_value = 1,
        max_value = 20,
        value     = 10)

    st.divider()

    # Health check
    health   = get_health()
    is_ok    = health.get(
        'status') == 'healthy'
    cache_s  = get_cache_stats()
    hit_rate = cache_s.get(
        'hit_rate_pct', 0)

    st.markdown("""
<div class="section-eyebrow">
  System Status
</div>
""", unsafe_allow_html=True)

    if is_ok:
        st.markdown("""
<div class="pill-ok">
  <span class="pulse-dot"></span>
  ALL SYSTEMS OPERATIONAL
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown("""
<div class="pill-warn">
  ⚠ DEGRADED
</div>
""", unsafe_allow_html=True)

    st.markdown(f"""
<div style="margin-top:16px;">
  <div style="display:flex;
              justify-content:space-between;
              padding:8px 0;
              border-bottom:1px solid #2a2a2a;">
    <span style="font-family:'JetBrains Mono',
                 monospace;font-size:10px;
                 color:#555;">
      CACHE HIT
    </span>
    <span style="font-family:'Bebas Neue',
                 sans-serif;font-size:16px;
                 color:#f5a623;">
      {hit_rate}%
    </span>
  </div>
  <div style="display:flex;
              justify-content:space-between;
              padding:8px 0;
              border-bottom:1px solid #2a2a2a;">
    <span style="font-family:'JetBrains Mono',
                 monospace;font-size:10px;
                 color:#555;">
      CACHED USERS
    </span>
    <span style="font-family:'Bebas Neue',
                 sans-serif;font-size:16px;
                 color:#f5a623;">
      {cache_s.get('cached_users', 0)}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()
    st.markdown("""
<div class="section-eyebrow">Quick Links</div>
""", unsafe_allow_html=True)

    links = {
        "API Docs":   "http://localhost:8000/docs",
        "Prometheus": "http://localhost:9090",
        "Grafana":    "http://localhost:3000",
        "MLflow":     "http://localhost:5000",
    }
    for name, url in links.items():
        st.markdown(
            f'<a href="{url}" target="_blank" '
            f'style="font-family:JetBrains Mono,'
            f'monospace;font-size:11px;'
            f'color:#a09890;text-decoration:none;'
            f'letter-spacing:0.1em;'
            f'display:block;padding:4px 0;">'
            f'→ {name}</a>',
            unsafe_allow_html=True)


# ── Hero Header ───────────────────────────────────
st.markdown(f"""
<div class="fade-up" style="padding:32px 0 8px">
  <div class="hero-sub">
    HSTU · META MLPERF 2026
  </div>
  <div class="hero-title">
    CINE<span style="color:#f5a623;">REC</span>
  </div>
  <div style="font-family:'Inter',sans-serif;
              font-size:14px;
              color:#a09890;
              margin-top:8px;
              max-width:520px;
              line-height:1.6;">
    Production-grade neural recommendation engine.
    Retrieval → Ranking → Re-ranking,
    served at scale.
  </div>
  <div class="amber-line"></div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "RECOMMENDATIONS",
    "WATCH HISTORY",
    "SYSTEM",
])


# ── Tab 1: Recommendations ────────────────────────
with tab1:
    # Top bar
    top_col1, top_col2, top_col3 = \
        st.columns([4, 2, 1])

    with top_col1:
        st.markdown(f"""
<div class="section-eyebrow">
  Personalised for User {user_id}
</div>
""", unsafe_allow_html=True)

    with top_col3:
        refresh = st.button(
            "↺ Refresh",
            use_container_width=True)

    # Fetch recs
    with st.spinner(""):
        start    = time.time()
        rec_data = get_recommendations(
            user_id, top_k)
        elapsed  = (time.time()-start)*1000

    if "error" in rec_data:
        st.markdown(f"""
<div style="background:#1a0a0a;
            border:1px solid #5a1a1a;
            border-radius:8px;
            padding:20px;
            font-family:'JetBrains Mono',
            monospace;font-size:12px;
            color:#f87171;">
  ⚠ API UNAVAILABLE — {rec_data['error']}<br><br>
  <span style="color:#555;">
    Start FastAPI:
    uvicorn src.serving.fastapi_app:app
    --port 8000
  </span>
</div>
""", unsafe_allow_html=True)
    else:
        recs   = rec_data.get(
            'recommendations', [])
        cached = rec_data.get(
            'cached', False)
        lat    = rec_data.get(
            'latency_ms', 0)

        # Meta strip
        meta_html = f"""
<div style="display:flex;gap:16px;
            align-items:center;
            margin-bottom:20px;
            flex-wrap:wrap;">
  <div class="{'pill-cached' if cached else 'pill-ok'}">
    {'⚡ REDIS CACHE' if cached else '🤖 HSTU MODEL'}
  </div>
  <span style="font-family:'JetBrains Mono',
               monospace;font-size:11px;
               color:#555;letter-spacing:0.1em;">
    {lat:.0f}ms
  </span>
  <span style="font-family:'JetBrains Mono',
               monospace;font-size:11px;
               color:#555;letter-spacing:0.1em;">
    {len(recs)} RESULTS
  </span>
  <span style="font-family:'JetBrains Mono',
               monospace;font-size:11px;
               color:#555;letter-spacing:0.1em;">
    RID {rec_data.get('request_id','—')}
  </span>
</div>
"""
        st.markdown(meta_html,
                    unsafe_allow_html=True)

        if not recs:
            st.markdown("""
<div style="text-align:center;padding:60px;
            color:#555;font-family:'JetBrains Mono',
            monospace;font-size:12px;
            letter-spacing:0.2em;">
  NO RECOMMENDATIONS FOUND
</div>
""", unsafe_allow_html=True)
        else:
            # Grid — 5 per row
            cols_per_row = 5
            for row_start in range(
                    0, len(recs),
                    cols_per_row):
                row_recs = recs[
                    row_start:
                    row_start+cols_per_row]
                cols = st.columns(
                    len(row_recs))

                for col, rec in zip(
                        cols, row_recs):
                    with col:
                        mid   = rec['movie_id']
                        title = str(rec.get(
                            'title',
                            f'Movie {mid}'))
                        score = float(rec.get(
                            'score', 0))
                        rank  = rec.get(
                            'rank', 0)
                        is_fb = rec.get(
                            'fallback', False)

                        # Get poster
                        m_row = movies[
                            movies['movieId']
                            == mid
                        ] if not movies.empty \
                            else pd.DataFrame()

                        poster_url = None
                        if not m_row.empty:
                            poster_url = \
                                get_poster_url(
                                    m_row.iloc[0])

                        badge = (
                            '<span class="pop-badge">'
                            'POPULAR</span>'
                            if is_fb else
                            '<span class="hstu-badge">'
                            'HSTU</span>')

                        if poster_url:
                            st.markdown(
                                f'<div class="movie-card">'
                                f'<span class="rank-badge">'
                                f'#{rank}</span>'
                                f'{badge}'
                                f'</div>',
                                unsafe_allow_html=True)
                            st.image(
                                poster_url,
                                use_column_width=True)
                        else:
                            st.markdown(
                                f"""
<div class="movie-card">
  <span class="rank-badge">#{rank}</span>
  {badge}
  <div class="movie-poster-placeholder">🎬</div>
</div>
""", unsafe_allow_html=True)

                        # Score bar
                        pct = min(
                            int(score*100), 100)
                        st.markdown(f"""
<div class="movie-info">
  <div class="movie-title-card">
    {title[:40]}
  </div>
  <div class="score-bar-wrap">
    <div class="score-bar-fill"
         style="width:{pct}%"></div>
  </div>
  <div class="score-label">
    SCORE {score:.3f}
  </div>
</div>
""", unsafe_allow_html=True)

                        # Rating expander
                        with st.expander(
                                f"Rate"):
                            stars = \
                                st.select_slider(
                                    f"s{mid}",
                                    options=[
                                        "1 ★",
                                        "2 ★★",
                                        "3 ★★★",
                                        "4 ★★★★",
                                        "5 ★★★★★"],
                                    label_visibility=
                                    "collapsed")
                            if st.button(
                                    "Submit",
                                    key=f"b{mid}"):
                                rating = int(
                                    stars[0])
                                ok = send_feedback(
                                    user_id,
                                    mid,
                                    float(rating))
                                if ok:
                                    st.success(
                                        "✓ Saved")
                                    st.cache_data\
                                        .clear()


# ── Tab 2: Watch History ──────────────────────────
with tab2:
    st.markdown(f"""
<div class="section-eyebrow">
  Watch History — User {user_id}
</div>
""", unsafe_allow_html=True)

    if ratings.empty:
        st.warning("No ratings data loaded")
    else:
        user_ratings = ratings[
            ratings['userId'] == user_id
        ].copy()

        if user_ratings.empty:
            st.markdown("""
<div style="text-align:center;
            padding:60px;
            color:#555;
            font-family:'JetBrains Mono',
            monospace;font-size:12px;
            letter-spacing:0.2em;">
  NO HISTORY FOR THIS USER
</div>
""", unsafe_allow_html=True)
        else:
            if not movies.empty:
                user_ratings = user_ratings\
                    .merge(
                        movies[[
                            'movieId',
                            'title',
                            'genres_list',
                            'year']],
                        on='movieId',
                        how='left')

            user_ratings['title'] = \
                user_ratings['title'].fillna(
                    user_ratings['movieId']\
                    .astype(str).apply(
                        lambda x: f"Movie {x}"))

            user_ratings = user_ratings\
                .sort_values(
                    'rating', ascending=False)

            # Stats row
            avg_r = user_ratings[
                'rating'].mean()
            n_r   = len(user_ratings)
            top_r = (user_ratings[
                'rating'] >= 4.0).sum()

            s1, s2, s3 = st.columns(3)
            for col, val, lbl in [
                (s1, n_r,           "MOVIES RATED"),
                (s2, f"{avg_r:.1f}", "AVG RATING"),
                (s3, top_r,          "TOP RATED (≥4)"),
            ]:
                with col:
                    st.markdown(f"""
<div class="stat-card">
  <div class="stat-value">{val}</div>
  <div class="stat-label">{lbl}</div>
</div>
""", unsafe_allow_html=True)

            st.markdown(
                "<div style='height:20px'></div>",
                unsafe_allow_html=True)

            # Top 10 list
            st.markdown("""
<div class="section-eyebrow">
  Top Rated
</div>
""", unsafe_allow_html=True)

            top10 = user_ratings.head(10)
            rows_html = ""
            for _, row in top10.iterrows():
                title  = str(
                    row.get('title', ''))[:45]
                rating = float(row['rating'])
                stars  = "★" * int(rating) + \
                          "☆" * (5-int(rating))
                year   = row.get('year', '')
                year_s = f" · {int(year)}" \
                    if pd.notna(year) \
                    and year != '' else ""

                rows_html += f"""
<div class="history-row">
  <div>
    <div class="history-title">
      {title}
      <span style="font-size:11px;
                   color:#555;">{year_s}</span>
    </div>
  </div>
  <div class="history-rating">
    {stars} {rating:.1f}
  </div>
</div>
"""
            st.markdown(
                f'<div style="background:'
                f'var(--surface);border:'
                f'1px solid var(--border);'
                f'border-radius:8px;'
                f'overflow:hidden;">'
                f'{rows_html}</div>',
                unsafe_allow_html=True)

            st.markdown(
                "<div style='height:24px'></div>",
                unsafe_allow_html=True)

            # Rating distribution
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("""
<div class="section-eyebrow">
  Rating Distribution
</div>
""", unsafe_allow_html=True)
                dist = user_ratings[
                    'rating']\
                    .value_counts()\
                    .sort_index()
                st.bar_chart(
                    dist,
                    color="#f5a623")

            with col_b:
                st.markdown("""
<div class="section-eyebrow">
  Genre Preferences
</div>
""", unsafe_allow_html=True)
                genre_counts = {}
                for _, row in \
                        user_ratings.iterrows():
                    for g in safe_genres(
                            row.get(
                                'genres_list')):
                        genre_counts[g] = \
                            genre_counts.get(
                                g, 0) + 1

                if genre_counts:
                    gdf = pd.DataFrame(
                        list(genre_counts.items()),
                        columns=[
                            'Genre', 'Count']
                    ).sort_values(
                        'Count',
                        ascending=False
                    ).head(8)
                    st.bar_chart(
                        gdf.set_index('Genre'),
                        color="#f5a623")


# ── Tab 3: System ─────────────────────────────────
with tab3:
    if st.button("↺ Refresh",
                 key="refresh_metrics"):
        st.cache_data.clear()

    m = get_metrics()
    c = get_cache_stats()

    st.markdown("""
<div class="section-eyebrow">
  Performance
</div>
""", unsafe_allow_html=True)

    mc = st.columns(4)
    for col, val, lbl in [
        (mc[0],
         m.get('total_requests', 0),
         "TOTAL REQUESTS"),
        (mc[1],
         f"{m.get('error_rate', 0):.1f}%",
         "ERROR RATE"),
        (mc[2],
         f"{m.get('latency_p50_ms', 0):.0f}ms",
         "P50 LATENCY"),
        (mc[3],
         f"{m.get('latency_p99_ms', 0):.0f}ms",
         "P99 LATENCY"),
    ]:
        with col:
            st.markdown(f"""
<div class="stat-card">
  <div class="stat-value">{val}</div>
  <div class="stat-label">{lbl}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        "<div style='height:24px'></div>",
        unsafe_allow_html=True)

    # Cache + services
    cc = st.columns(2)
    with cc[0]:
        st.markdown("""
<div class="section-eyebrow">
  Cache
</div>
""", unsafe_allow_html=True)
        hit  = c.get('hit_rate_pct', 0)
        ccon = c.get('connected', False)
        st.markdown(f"""
<div class="stat-card" style="text-align:left;
     padding:20px;">
  <div style="display:flex;justify-content:
  space-between;margin-bottom:12px;">
    <span style="font-family:'JetBrains Mono',
    monospace;font-size:10px;color:#555;
    letter-spacing:0.2em;">REDIS</span>
    <span class="{'pill-ok' if ccon else 'pill-warn'}">
      {'CONNECTED' if ccon else 'DOWN'}
    </span>
  </div>
  <div class="stat-value">{hit}%</div>
  <div class="stat-label">HIT RATE</div>
  <div style="background:#1f1f1f;height:4px;
  border-radius:2px;margin-top:12px;">
    <div style="background:linear-gradient(
    90deg,#c47f10,#f5a623);height:100%;
    width:{hit}%;border-radius:2px;">
    </div>
  </div>
  <div style="margin-top:12px;font-family:
  'JetBrains Mono',monospace;font-size:10px;
  color:#555;">
    {c.get('cached_users', 0)} users cached ·
    {c.get('memory_used', 'N/A')} memory
  </div>
</div>
""", unsafe_allow_html=True)

    with cc[1]:
        st.markdown("""
<div class="section-eyebrow">
  Services
</div>
""", unsafe_allow_html=True)

        services = [
            ("FASTAPI",    8000,
             True),
            ("BENTOML",    3001,
             True),
            ("REDIS",      6379,
             m.get('redis_connected',
                   False)),
            ("KAFKA",      9092,
             m.get('kafka_connected',
                   False)),
            ("PROMETHEUS", 9090,
             True),
            ("GRAFANA",    3000,
             True),
        ]

        rows = ""
        for svc, port, ok in services:
            dot   = "#4ade80" if ok else "#f87171"
            color = "#f0ede8" if ok else "#555"
            rows += f"""
<div style="display:flex;justify-content:
space-between;align-items:center;
padding:10px 16px;
border-bottom:1px solid #2a2a2a;">
  <div style="display:flex;
  align-items:center;gap:10px;">
    <div style="width:6px;height:6px;
    border-radius:50%;
    background:{dot};"></div>
    <span style="font-family:'JetBrains Mono',
    monospace;font-size:11px;
    color:{color};
    letter-spacing:0.12em;">{svc}</span>
  </div>
  <span style="font-family:'JetBrains Mono',
  monospace;font-size:10px;
  color:#555;">:{port}</span>
</div>
"""
        st.markdown(
            f'<div style="background:'
            f'var(--surface);border:'
            f'1px solid var(--border);'
            f'border-radius:8px;'
            f'overflow:hidden;">'
            f'{rows}</div>',
            unsafe_allow_html=True)

    st.markdown(
        "<div style='height:24px'></div>",
        unsafe_allow_html=True)

    # Architecture
    st.markdown("""
<div class="section-eyebrow">
  Architecture
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="arch-box">
  <span style="color:#f5a623;">USER BROWSER</span>
  <br>↓
  <br><span style="color:#f0ede8;">
    STREAMLIT :8501
  </span>
  <span style="color:#555;">
    — Frontend UI
  </span>
  <br>↓
  <br><span style="color:#f0ede8;">
    FASTAPI :8000
  </span>
  <span style="color:#555;">
    — Rate limit · Auth · CORS
  </span>
  <br>↓ cache hit &nbsp;&nbsp;&nbsp;&nbsp; ↓ miss
  <br><span style="color:#4ade80;">
    REDIS :6379
  </span>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <span style="color:#60a5fa;">
    BENTOML :3001
  </span>
  <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  ↓
  <br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <span style="color:#60a5fa;">HSTU MODEL</span>
  <br>↓
  <br><span style="color:#f0ede8;">
    KAFKA :9092
  </span>
  <span style="color:#555;">
    — Event streaming
  </span>
  <br>↓
  <br><span style="color:#f0ede8;">
    PROMETHEUS :9090 → GRAFANA :3000
  </span>
</div>
""", unsafe_allow_html=True)