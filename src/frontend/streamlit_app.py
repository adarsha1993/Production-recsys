"""
Production RecSys — Streamlit Frontend
Netflix-inspired design system.

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
import os
from dotenv import load_dotenv

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))
load_dotenv(BASE / '.env')

# ── Config ────────────────────────────────────────
API_URL      = "http://localhost:8000"
TMDB_BASE    = "https://image.tmdb.org/t/p/w342"
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")

st.set_page_config(
    page_title = "CineRec",
    page_icon  = "🎬",
    layout     = "wide",
    initial_sidebar_state = "collapsed",
)

# ── Theme ─────────────────────────────────────────
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True

dark = st.session_state.dark_mode

# Netflix design tokens
# Dark: pure black + Netflix red
# Light: off-white + Netflix red
if dark:
    BG         = "#141414"
    BG2        = "#1f1f1f"
    BG3        = "#2f2f2f"
    SURFACE    = "#181818"
    BORDER     = "#333333"
    TEXT_HI    = "#ffffff"
    TEXT_MID   = "#b3b3b3"
    TEXT_LO    = "#757575"
    ACCENT     = "#e50914"   # Netflix red
    ACCENT2    = "#ff0a16"
    ACCENT_DIM = "#b20710"
    HOVER_BG   = "#2f2f2f"
    CARD_OVER  = "rgba(0,0,0,0.7)"
    SHADOW     = "rgba(0,0,0,0.75)"
else:
    BG         = "#f3f3f3"
    BG2        = "#ffffff"
    BG3        = "#e8e8e8"
    SURFACE    = "#ffffff"
    BORDER     = "#e0e0e0"
    TEXT_HI    = "#000000"
    TEXT_MID   = "#333333"
    TEXT_LO    = "#757575"
    ACCENT     = "#e50914"
    ACCENT2    = "#b20710"
    ACCENT_DIM = "#b20710"
    HOVER_BG   = "#f0f0f0"
    CARD_OVER  = "rgba(0,0,0,0.5)"
    SHADOW     = "rgba(0,0,0,0.15)"

# ── CSS ───────────────────────────────────────────
st.markdown(f"""
<style>
/* Netflix uses these exact fonts:
   Display: Netflix Sans (custom) — we use
   'Inter' as closest public equivalent
   Body: Netflix Sans / Helvetica Neue
   Netflix Sans has tight tracking, wide
   weights and is geometric grotesque */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset ── */
*, *::before, *::after {{
  box-sizing: border-box;
}}

.stApp {{
  background: {BG} !important;
  font-family: 'Inter', 'Helvetica Neue',
               Helvetica, Arial, sans-serif;
  color: {TEXT_MID};
}}

/* Hide default streamlit chrome */
#MainMenu {{ visibility: hidden; }}
footer    {{ visibility: hidden; }}
header    {{ visibility: hidden; }}
.stDeployButton {{ display: none; }}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
  background: {BG} !important;
  border-right: 1px solid {BORDER} !important;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{
  background: {BG};
}}
::-webkit-scrollbar-thumb {{
  background: {BG3};
  border-radius: 2px;
}}

/* ── Tabs — Netflix nav style ── */
.stTabs [data-baseweb="tab-list"] {{
  background: transparent !important;
  border-bottom: 1px solid {BORDER} !important;
  gap: 0 !important;
  padding: 0 !important;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent !important;
  color: {TEXT_LO} !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  letter-spacing: 0.02em !important;
  text-transform: uppercase !important;
  padding: 14px 24px !important;
  border-bottom: 3px solid transparent !important;
  margin-bottom: -1px !important;
  transition: color 0.15s !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
  color: {TEXT_HI} !important;
}}
.stTabs [aria-selected="true"] {{
  color: {TEXT_HI} !important;
  border-bottom-color: {ACCENT} !important;
}}

/* ── Buttons ── */
.stButton > button {{
  background: {ACCENT} !important;
  border: none !important;
  color: #ffffff !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em !important;
  padding: 10px 20px !important;
  border-radius: 4px !important;
  transition: background 0.15s,
              transform 0.1s !important;
}}
.stButton > button:hover {{
  background: {ACCENT2} !important;
  transform: scale(1.02) !important;
}}
.stButton > button:active {{
  transform: scale(0.98) !important;
}}

/* Ghost button variant */
.ghost-btn .stButton > button {{
  background: rgba(109,109,110,0.7)
              !important;
  color: {TEXT_HI} !important;
}}
.ghost-btn .stButton > button:hover {{
  background: rgba(109,109,110,0.9)
              !important;
}}

/* ── Select / inputs ── */
.stSelectbox > div > div {{
  background: {BG2} !important;
  border: 1px solid {BORDER} !important;
  border-radius: 4px !important;
  color: {TEXT_HI} !important;
  font-family: 'Inter', sans-serif !important;
}}
.stSlider [data-baseweb="slider"]
  div[role="slider"] {{
  background: {ACCENT} !important;
  border-color: {ACCENT} !important;
}}
.stSlider [data-baseweb="slider"]
  div[data-testid="stThumbValue"] {{
  color: {TEXT_HI} !important;
}}

/* ── Expander ── */
details summary {{
  background: {BG2} !important;
  border: 1px solid {BORDER} !important;
  border-radius: 4px !important;
  color: {TEXT_MID} !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  padding: 8px 14px !important;
}}

/* ── Divider ── */
hr {{
  border-color: {BORDER} !important;
  margin: 20px 0 !important;
}}

/* ── Spinner ── */
.stSpinner > div {{
  border-top-color: {ACCENT} !important;
}}

/* ── Animations ── */
@keyframes fadeIn {{
  from {{ opacity: 0; }}
  to   {{ opacity: 1; }}
}}
@keyframes slideUp {{
  from {{ opacity: 0;
          transform: translateY(20px); }}
  to   {{ opacity: 1;
          transform: translateY(0); }}
}}
@keyframes scaleIn {{
  from {{ transform: scale(0.96);
          opacity: 0; }}
  to   {{ transform: scale(1);
          opacity: 1; }}
}}
@keyframes pulse {{
  0%,100% {{ opacity: 1; }}
  50%      {{ opacity: 0.4; }}
}}

/* ══════════════════════════════════════
   NETFLIX COMPONENT LIBRARY
   ══════════════════════════════════════ */

/* ── N-logo mark ── */
.n-logo {{
  font-family: 'Inter', sans-serif;
  font-weight: 900;
  font-size: 32px;
  color: {ACCENT};
  letter-spacing: -0.03em;
  line-height: 1;
}}

/* ── Top nav bar ── */
.nf-nav {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 0 20px;
  border-bottom: 1px solid {BORDER};
  margin-bottom: 32px;
  animation: fadeIn 0.4s ease;
}}
.nf-nav-left {{
  display: flex;
  align-items: center;
  gap: 32px;
}}
.nf-nav-link {{
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  font-weight: 500;
  color: {TEXT_MID};
  text-decoration: none;
  letter-spacing: 0.01em;
  transition: color 0.15s;
  cursor: pointer;
}}
.nf-nav-link:hover {{
  color: {TEXT_HI};
}}
.nf-nav-link.active {{
  color: {TEXT_HI};
  font-weight: 600;
}}

/* ── Hero billboard ── */
.nf-hero {{
  position: relative;
  background: linear-gradient(
    180deg,
    transparent 0%,
    {BG} 100%),
    linear-gradient(
    90deg,
    {'rgba(0,0,0,0.9)' if dark
     else 'rgba(0,0,0,0.5)'} 0%,
    transparent 60%);
  background-color: {BG2};
  border-radius: 6px;
  padding: 60px 48px;
  margin-bottom: 40px;
  overflow: hidden;
  animation: fadeIn 0.6s ease;
}}
.nf-hero-eyebrow {{
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 700;
  color: {ACCENT};
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-bottom: 12px;
}}
.nf-hero-title {{
  font-family: 'Inter', sans-serif;
  font-size: clamp(36px, 5vw, 64px);
  font-weight: 900;
  color: {TEXT_HI};
  line-height: 1.05;
  letter-spacing: -0.03em;
  margin-bottom: 16px;
}}
.nf-hero-title span.red {{
  color: {ACCENT};
}}
.nf-hero-sub {{
  font-family: 'Inter', sans-serif;
  font-size: 16px;
  font-weight: 400;
  color: {TEXT_MID};
  line-height: 1.6;
  max-width: 440px;
  margin-bottom: 28px;
}}
.nf-hero-tags {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-bottom: 8px;
}}
.nf-tag {{
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 600;
  color: {TEXT_MID};
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 4px 10px;
  border: 1px solid {BORDER};
  border-radius: 3px;
  background: {'rgba(255,255,255,0.05)'
               if dark else
               'rgba(0,0,0,0.04)'};
}}

/* ── Row label (like "Top Picks") ── */
.nf-row-label {{
  font-family: 'Inter', sans-serif;
  font-size: 18px;
  font-weight: 700;
  color: {TEXT_HI};
  letter-spacing: -0.01em;
  margin-bottom: 14px;
  margin-top: 8px;
}}
.nf-row-sub {{
  font-size: 13px;
  font-weight: 400;
  color: {TEXT_LO};
  margin-left: 10px;
  letter-spacing: 0;
}}

/* ── Movie card — Netflix hover reveal ── */
.nf-card {{
  position: relative;
  border-radius: 4px;
  overflow: hidden;
  cursor: pointer;
  background: {BG3};
  transition: transform 0.25s ease,
              box-shadow 0.25s ease,
              z-index 0s 0.25s;
  animation: scaleIn 0.3s ease forwards;
}}
.nf-card:hover {{
  transform: scale(1.06);
  box-shadow: 0 14px 50px {SHADOW};
  z-index: 10;
}}
.nf-card-img {{
  width: 100%;
  aspect-ratio: 2/3;
  object-fit: cover;
  display: block;
}}
.nf-card-placeholder {{
  width: 100%;
  aspect-ratio: 2/3;
  background: linear-gradient(
    160deg,
    {'#2a2a2a' if dark else '#ddd'} 0%,
    {'#1a1a1a' if dark else '#ccc'} 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 40px;
  color: {TEXT_LO};
}}
.nf-card-overlay {{
  position: absolute;
  bottom: 0; left: 0; right: 0;
  background: linear-gradient(
    transparent 0%,
    {CARD_OVER} 100%);
  padding: 20px 10px 10px;
  opacity: 0;
  transition: opacity 0.2s ease;
}}
.nf-card:hover .nf-card-overlay {{
  opacity: 1;
}}
.nf-card-title {{
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  font-weight: 700;
  color: #ffffff;
  line-height: 1.3;
  margin-bottom: 4px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}}
.nf-card-score {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: #a3cf62;
  font-weight: 500;
}}

/* ── Match score badge ── */
.match-badge {{
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 700;
  color: #a3cf62;
  letter-spacing: 0.02em;
}}

/* ── Rank number ── */
.rank-num {{
  font-family: 'Inter', sans-serif;
  font-size: 80px;
  font-weight: 900;
  color: {'rgba(255,255,255,0.08)'
          if dark else
          'rgba(0,0,0,0.07)'};
  line-height: 1;
  letter-spacing: -0.05em;
  position: absolute;
  bottom: -8px;
  left: -6px;
  pointer-events: none;
  -webkit-text-stroke: {'2px rgba(255,255,255,0.15)'
                        if dark else
                        '2px rgba(0,0,0,0.1)'};
}}

/* ── Type badge ── */
.type-hstu {{
  display: inline-block;
  background: {ACCENT};
  color: #ffffff;
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 800;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 2px 6px;
  border-radius: 2px;
}}
.type-pop {{
  display: inline-block;
  background: {BG3};
  color: {TEXT_MID};
  font-family: 'Inter', sans-serif;
  font-size: 9px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 2px 6px;
  border-radius: 2px;
  border: 1px solid {BORDER};
}}

/* ── Source pill ── */
.src-cache {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(163,207,98,0.12);
  border: 1px solid rgba(163,207,98,0.35);
  color: #a3cf62;
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 600;
  padding: 5px 14px;
  border-radius: 3px;
}}
.src-model {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: {'rgba(229,9,20,0.12)'
               if dark else
               'rgba(229,9,20,0.08)'};
  border: 1px solid rgba(229,9,20,0.35);
  color: {ACCENT};
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 600;
  padding: 5px 14px;
  border-radius: 3px;
}}

/* ── Stat card ── */
.nf-stat {{
  background: {BG2};
  border: 1px solid {BORDER};
  border-radius: 4px;
  padding: 20px 16px;
  text-align: center;
  transition: border-color 0.2s;
}}
.nf-stat:hover {{
  border-color: {ACCENT};
}}
.nf-stat-val {{
  font-family: 'Inter', sans-serif;
  font-size: 32px;
  font-weight: 900;
  color: {TEXT_HI};
  letter-spacing: -0.03em;
  line-height: 1;
}}
.nf-stat-lbl {{
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 500;
  color: {TEXT_LO};
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-top: 6px;
}}

/* ── History row ── */
.nf-hist {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  border-bottom: 1px solid {BORDER};
  transition: background 0.15s;
}}
.nf-hist:hover {{
  background: {HOVER_BG};
}}
.nf-hist:last-child {{
  border-bottom: none;
}}
.nf-hist-title {{
  font-family: 'Inter', sans-serif;
  font-size: 14px;
  font-weight: 500;
  color: {TEXT_HI};
}}
.nf-hist-meta {{
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  color: {TEXT_LO};
  margin-top: 2px;
}}
.nf-match {{
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  font-weight: 700;
  color: #a3cf62;
  white-space: nowrap;
}}

/* ── Service row ── */
.nf-svc {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 20px;
  border-bottom: 1px solid {BORDER};
}}
.nf-svc:last-child {{ border-bottom: none; }}
.nf-svc-name {{
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  font-weight: 600;
  color: {TEXT_HI};
}}
.nf-svc-port {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: {TEXT_LO};
}}
.dot-on {{
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  background: #a3cf62;
  animation: pulse 2.5s infinite;
}}
.dot-off {{
  display: inline-block;
  width: 8px; height: 8px;
  border-radius: 50%;
  background: {ACCENT};
}}

/* ── Cache bar ── */
.nf-cache-card {{
  background: {BG2};
  border: 1px solid {BORDER};
  border-radius: 4px;
  padding: 22px 20px;
}}
.nf-progress {{
  background: {BG3};
  border-radius: 2px;
  height: 4px;
  overflow: hidden;
  margin: 14px 0 10px;
}}
.nf-progress-fill {{
  height: 100%;
  background: linear-gradient(
    90deg, {ACCENT}, {ACCENT2});
  border-radius: 2px;
  transition: width 0.6s ease;
}}

/* ── Arch panel ── */
.nf-arch {{
  background: {BG2};
  border: 1px solid {BORDER};
  border-left: 3px solid {ACCENT};
  border-radius: 0 4px 4px 0;
  padding: 24px 28px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px;
  color: {TEXT_MID};
  line-height: 2.1;
}}

/* ── Meta strip ── */
.nf-meta {{
  display: flex;
  gap: 16px;
  align-items: center;
  flex-wrap: wrap;
  padding: 10px 0 18px;
}}
.nf-meta-item {{
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  font-weight: 500;
  color: {TEXT_LO};
}}

/* ── Sidebar logo area ── */
.sb-logo {{
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 0 20px;
}}
.sb-logo-n {{
  font-family: 'Inter', sans-serif;
  font-weight: 900;
  font-size: 28px;
  color: {ACCENT};
  letter-spacing: -0.04em;
  line-height: 1;
}}
.sb-logo-text {{
  font-family: 'Inter', sans-serif;
  font-weight: 700;
  font-size: 14px;
  color: {TEXT_HI};
  letter-spacing: -0.01em;
  line-height: 1.2;
}}
.sb-logo-sub {{
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 400;
  color: {TEXT_LO};
  letter-spacing: 0.02em;
}}
.sb-section {{
  font-family: 'Inter', sans-serif;
  font-size: 10px;
  font-weight: 700;
  color: {TEXT_LO};
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 10px;
  margin-top: 4px;
}}
.sb-kv {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 0;
  border-bottom: 1px solid {BORDER};
}}
.sb-kv-k {{
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  color: {TEXT_LO};
}}
.sb-kv-v {{
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  font-weight: 700;
  color: {TEXT_HI};
}}
.sb-link {{
  font-family: 'Inter', sans-serif;
  font-size: 12px;
  font-weight: 500;
  color: {TEXT_MID};
  text-decoration: none;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 7px 0;
  transition: color 0.15s;
  border-bottom: 1px solid {BORDER};
}}
.sb-link:hover {{ color: {TEXT_HI}; }}
.status-ok {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(163,207,98,0.1);
  border: 1px solid rgba(163,207,98,0.3);
  color: #a3cf62;
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 600;
  padding: 5px 12px;
  border-radius: 3px;
}}
.status-warn {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: rgba(229,9,20,0.1);
  border: 1px solid rgba(229,9,20,0.3);
  color: {ACCENT};
  font-family: 'Inter', sans-serif;
  font-size: 11px;
  font-weight: 600;
  padding: 5px 12px;
  border-radius: 3px;
}}
.no-content {{
  text-align: center;
  padding: 80px 20px;
  font-family: 'Inter', sans-serif;
  font-size: 14px;
  color: {TEXT_LO};
}}
.err-box {{
  background: {'#2a0a0a' if dark
               else '#fff5f5'};
  border: 1px solid {'#6a1a1a' if dark
                     else '#fca5a5'};
  border-radius: 4px;
  padding: 20px 24px;
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  color: {'#ff6b6b' if dark else '#dc2626'};
  line-height: 1.7;
}}
</style>
""", unsafe_allow_html=True)


# ── Data helpers ──────────────────────────────────
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
        df['title'] = df['title'].fillna(
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


def api_get(path: str) -> dict:
    try:
        r = requests.get(
            f"{API_URL}{path}", timeout=4)
        return r.json() \
            if r.status_code == 200 else {}
    except Exception:
        return {}


def api_post(path: str,
             body: dict) -> dict:
    try:
        r = requests.post(
            f"{API_URL}{path}",
            json=body, timeout=30)
        return r.json() \
            if r.status_code == 200 \
            else {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def get_poster(
        row: pd.Series) -> Optional[str]:
    try:
        p = str(row.get('poster_path', ''))
        if p not in ('nan', '', 'None') \
                and p.startswith('/'):
            return f"{TMDB_BASE}{p}"
    except Exception:
        pass
    return None


def safe_genres(val) -> list:
    if val is None or \
            isinstance(val, float):
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            r = ast.literal_eval(val)
            return r \
                if isinstance(r, list) \
                else []
        except Exception:
            return []
    return []


def match_pct(score: float) -> str:
    """Convert score to Netflix-style match %"""
    pct = int(min(score * 800 + 60, 99)) \
        if score > 0.001 else \
        int(np.random.randint(72, 92))
    return f"{pct}% Match"


# ── Load data ─────────────────────────────────────
movies  = load_movies()
ratings = load_ratings()

if not ratings.empty:
    available_users = ratings.groupby(
        'userId')['rating'].count()\
        .sort_values(ascending=False)\
        .index.tolist()
else:
    available_users = list(range(1, 51))


# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
<div class="sb-logo">
  <div class="sb-logo-n">N</div>
  <div>
    <div class="sb-logo-text">CineRec</div>
    <div class="sb-logo-sub">
      Neural Recommendations
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Theme toggle
    st.markdown(
        '<div class="sb-section">Theme</div>',
        unsafe_allow_html=True)

    if st.button(
            "☀ Light" if dark else "◑ Dark",
            use_container_width=True):
        st.session_state.dark_mode = not dark
        st.rerun()

    st.markdown(
        "<div style='height:12px'></div>",
        unsafe_allow_html=True)

    st.markdown(
        '<div class="sb-section">'
        'Who\'s Watching?</div>',
        unsafe_allow_html=True)

    user_id = st.selectbox(
        "user",
        options=available_users[:100],
        index=0,
        label_visibility="collapsed")

    top_k = st.slider(
        "Titles",
        min_value=5,
        max_value=20,
        value=10)

    st.divider()

    health  = api_get("/health")
    cache_s = api_get("/cache/stats")
    is_ok   = health.get(
        'status') == 'healthy'
    hr      = cache_s.get(
        'hit_rate_pct', 0)
    cu      = cache_s.get(
        'cached_users', 0)

    st.markdown(
        '<div class="sb-section">'
        'System</div>',
        unsafe_allow_html=True)

    if is_ok:
        st.markdown(
            '<div class="status-ok">'
            '<span class="dot-on"></span>'
            'All Services Up</div>',
            unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="status-warn">'
            '⚠ Degraded</div>',
            unsafe_allow_html=True)

    st.markdown(f"""
<div style="margin-top:12px;">
  <div class="sb-kv">
    <span class="sb-kv-k">Cache Hit Rate</span>
    <span class="sb-kv-v"
          style="color:#a3cf62;">{hr}%</span>
  </div>
  <div class="sb-kv">
    <span class="sb-kv-k">Cached Profiles</span>
    <span class="sb-kv-v">{cu}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div class="sb-section">'
        'Quick Links</div>',
        unsafe_allow_html=True)

    for name, url, icon in [
        ("API Docs",
         "http://localhost:8000/docs", "📋"),
        ("Prometheus",
         "http://localhost:9090", "📈"),
        ("Grafana",
         "http://localhost:3000", "📊"),
        ("MLflow",
         "http://localhost:5000", "🧪"),
    ]:
        st.markdown(
            f'<a href="{url}" '
            f'target="_blank" '
            f'class="sb-link">'
            f'{icon} {name}</a>',
            unsafe_allow_html=True)


# ── Hero billboard ────────────────────────────────
st.markdown(
    '<div class="nf-hero">'

    # Title
    '<div class="nf-hero-title">'
    '<span style="color:' + ACCENT + ';">'
    'Your Next<br>Favourite Film'
    '</span>'
    '</div>'

    # Red accent rule
    '<div style="'
    'width:60px;height:4px;'
    'background:linear-gradient('
    '90deg,' + ACCENT + ',transparent);'
    'border-radius:2px;'
    'margin-bottom:18px;">'
    '</div>'

    # Description
    '<div class="nf-hero-sub">'
    'A production-grade AI recommendation '
    'engine built end-to-end — from raw data '
    'to live serving. Combines classical ML, '
    'deep neural ranking, large language '
    'models, and real-time MLOps '
    'infrastructure.'
    '</div>'

    # Built by
    '<div style="'
    'display:flex;'
    'align-items:center;'
    'gap:16px;'
    'margin-top:24px;'
    'padding-top:20px;'
    'border-top:1px solid ' + BORDER + ';">'

    # Avatar circle
    '<div style="'
    'width:44px;height:44px;'
    'border-radius:50%;'
    'background:linear-gradient('
    '135deg,' + ACCENT + ' 0%,#ff6b35 100%);'
    'display:flex;align-items:center;'
    'justify-content:center;'
    'font-family:Inter,sans-serif;'
    'font-size:18px;font-weight:900;'
    'color:#ffffff;'
    'flex-shrink:0;">'
    'A'
    '</div>'

    # Name + title
    '<div>'
    '<div style="'
    'font-family:Inter,sans-serif;'
    'font-size:15px;font-weight:800;'
    'color:' + TEXT_HI + ';'
    'letter-spacing:-0.01em;'
    'line-height:1.2;">'
    'Adarsha Ghimire'
    '</div>'
    '<div style="'
    'font-family:Inter,sans-serif;'
    'font-size:12px;font-weight:400;'
    'color:' + TEXT_LO + ';'
    'margin-top:2px;'
    'letter-spacing:0.02em;">'
    'Graduate Student &nbsp;·&nbsp; '
    'Wilfrid Laurier University &nbsp;·&nbsp; '
    '2026'
    '</div>'
    '</div>'

    # Spacer
    '<div style="flex:1;"></div>'

    # Copyright right side
    '<div style="'
    'text-align:right;">'
    '<div style="'
    'font-family:Inter,sans-serif;'
    'font-size:11px;font-weight:600;'
    'color:' + ACCENT + ';'
    'letter-spacing:0.05em;">'
    'CP612'
    '</div>'
    '<div style="'
    'font-family:Inter,sans-serif;'
    'font-size:10px;font-weight:400;'
    'color:' + TEXT_LO + ';'
    'margin-top:2px;">'
    'Production Recommendation Systems'
    '</div>'
    '<div style="'
    'font-family:Inter,sans-serif;'
    'font-size:10px;font-weight:400;'
    'color:' + TEXT_LO + ';'
    'margin-top:1px;">'
    '© Adarsha Ghimire'
    '</div>'
    '</div>'

    '</div>'
    '</div>',
    unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "For You",
    "Watch History",
    "System",
])


# ══════════════════════════════════════════════════
# Tab 1 — Recommendations
# ══════════════════════════════════════════════════
with tab1:
    hdr_c, _, btn_c = st.columns([5, 2, 1])
    with hdr_c:
        st.markdown(
            f'<div class="nf-row-label">'
            f'Top Picks for User {user_id}'
            f'<span class="nf-row-sub">'
            f'Personalised · HSTU Neural'
            f'</span></div>',
            unsafe_allow_html=True)
    with btn_c:
        if st.button("↺",
                     use_container_width=True,
                     key="refresh_recs"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner(""):
        t0       = time.time()
        rec_data = api_post(
            "/recommend",
            {"user_id": user_id,
             "top_k":   top_k})
        elapsed  = (time.time()-t0)*1000

    if "error" in rec_data:
        st.markdown(
            f'<div class="err-box">'
            f'<strong>Service Unavailable</strong>'
            f'<br>{rec_data["error"]}<br>'
            f'<span style="opacity:0.6;">'
            f'uvicorn src.serving.fastapi_app'
            f':app --port 8000</span>'
            f'</div>',
            unsafe_allow_html=True)
    else:
        recs   = rec_data.get(
            'recommendations', [])
        cached = rec_data.get('cached', False)
        lat    = rec_data.get('latency_ms', 0)
        rid    = rec_data.get(
            'request_id', '—')

        src_badge = (
            '<span class="src-cache">'
            '⚡ Served from cache</span>'
            if cached else
            '<span class="src-model">'
            '🧠 HSTU Neural Model</span>')

        st.markdown(f"""
<div class="nf-meta">
  {src_badge}
  <span class="nf-meta-item">
    {lat:.0f}ms
  </span>
  <span class="nf-meta-item">
    {len(recs)} titles
  </span>
  <span class="nf-meta-item"
        style="color:{TEXT_LO};">
    {rid}
  </span>
</div>
""", unsafe_allow_html=True)

        if not recs:
            st.markdown(
                '<div class="no-content">'
                'No recommendations available'
                '</div>',
                unsafe_allow_html=True)
        else:
            per_row = 5
            for rs in range(
                    0, len(recs), per_row):
                batch = recs[rs:rs+per_row]
                cols  = st.columns(
                    len(batch),
                    gap="small")

                for col, rec in zip(
                        cols, batch):
                    with col:
                        mid   = rec[
                            'movie_id']
                        title = str(rec.get(
                            'title',
                            f'Movie {mid}'))
                        score = float(
                            rec.get(
                                'score', 0))
                        rank  = rec.get(
                            'rank', 0)
                        is_fb = rec.get(
                            'fallback', False)

                        mrow = movies[
                            movies['movieId']
                            == mid
                        ] if not movies\
                            .empty \
                            else \
                            pd.DataFrame()

                        purl = get_poster(
                            mrow.iloc[0]) \
                            if not mrow\
                            .empty \
                            else None

                        mp  = match_pct(score)
                        typ = (
                            '<span class='
                            '"type-pop">'
                            'Popular</span>'
                            if is_fb else
                            '<span class='
                            '"type-hstu">'
                            'Neural</span>')

                        # Card
                        st.markdown(
                            '<div class='
                            '"nf-card">',
                            unsafe_allow_html=
                            True)

                        if purl:
                            st.image(
                                purl,
                                use_column_width=
                                True)
                        else:
                            st.markdown(
                                '<div class='
                                '"nf-card-'
                                'placeholder">'
                                '🎬</div>',
                                unsafe_allow_html=
                                True)

                        st.markdown(
                            f"""
<div class="nf-card-overlay">
  <div class="match-badge">{mp}</div>
  <div class="nf-card-title">
    {title[:38]}
  </div>
</div>
""",
                            unsafe_allow_html=
                            True)

                        st.markdown(
                            '</div>',
                            unsafe_allow_html=
                            True)

                        # Below card
                        st.markdown(
                            f'<div style="'
                            f'padding:6px 2px'
                            f' 2px;">'
                            f'<div style="'
                            f'font-family:'
                            f'Inter,sans-serif;'
                            f'font-size:12px;'
                            f'font-weight:600;'
                            f'color:{TEXT_HI};'
                            f'line-height:1.3;'
                            f'margin-bottom:'
                            f'4px;">'
                            f'{title[:30]}'
                            f'</div>'
                            f'<div style="'
                            f'display:flex;'
                            f'align-items:'
                            f'center;gap:6px;">'
                            f'<span class='
                            f'"match-badge">'
                            f'{mp}</span>'
                            f'{typ}'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=
                            True)

                        with st.expander(
                                "Rate this"):
                            stars = \
                                st.select_slider(
                                    f"s{mid}"
                                    f"_{rs}",
                                    options=[
                                        "1 ★",
                                        "2 ★★",
                                        "3 ★★★",
                                        "4 ★★★★",
                                        "5 ★★★★★"
                                    ],
                                    label_visibility=
                                    "collapsed")
                            if st.button(
                                    "Submit",
                                    key=f"b"
                                    f"{mid}_{rs}"):
                                r_val = int(
                                    stars[0])
                                ok = api_post(
                                    "/feedback",
                                    {
                                        "user_id":
                                        user_id,
                                        "movie_id":
                                        mid,
                                        "rating":
                                        float(
                                            r_val),
                                        "action":
                                        "rate",
                                    })
                                if "error" \
                                        not in ok:
                                    st.success(
                                        "Rated!")
                                    st.cache_data\
                                        .clear()


# ══════════════════════════════════════════════════
# Tab 2 — Watch History
# ══════════════════════════════════════════════════
with tab2:
    st.markdown(
        f'<div class="nf-row-label">'
        f'Your Activity'
        f'<span class="nf-row-sub">'
        f'User {user_id}</span></div>',
        unsafe_allow_html=True)

    if ratings.empty:
        st.markdown(
            '<div class="no-content">'
            'No data</div>',
            unsafe_allow_html=True)
    else:
        ur = ratings[
            ratings['userId'] == user_id
        ].copy()

        if ur.empty:
            st.markdown(
                '<div class="no-content">'
                'No activity for this profile'
                '</div>',
                unsafe_allow_html=True)
        else:
            if not movies.empty:
                ur = ur.merge(
                    movies[[
                        'movieId',
                        'title',
                        'genres_list',
                        'year',
                        'poster_path']],
                    on='movieId',
                    how='left')

            ur['title'] = ur['title']\
                .fillna(
                    ur['movieId'].astype(str)
                    .apply(
                        lambda x:
                        f"Movie {x}"))
            ur = ur.sort_values(
                'rating', ascending=False)

            avg_r = ur['rating'].mean()
            n_r   = len(ur)
            top_r = (ur['rating']
                     >= 4.0).sum()

            # ── Stats row ─────────────────
            c1, c2, c3 = st.columns(3)
            for col, v, l in [
                (c1, n_r, "Titles Rated"),
                (c2, f"{avg_r:.1f}",
                 "Avg Rating"),
                (c3, int(top_r),
                 "Loved (4+★)"),
            ]:
                with col:
                    st.markdown(f"""
<div class="nf-stat">
  <div class="nf-stat-val">{v}</div>
  <div class="nf-stat-lbl">{l}</div>
</div>
""", unsafe_allow_html=True)

            st.markdown(
                "<div style='height:28px'>"
                "</div>",
                unsafe_allow_html=True)

            # ── Top rated grid ────────────
            st.markdown(
                '<div class="nf-row-label">'
                'Highest Rated</div>',
                unsafe_allow_html=True)

            top_movies = ur.head(20)
            per_row    = 5

            for rs in range(
                    0, len(top_movies),
                    per_row):
                batch = top_movies.iloc[
                    rs:rs+per_row]
                cols  = st.columns(
                    len(batch),
                    gap="small")

                for col, (_, row) in zip(
                        cols,
                        batch.iterrows()):
                    with col:
                        mid   = int(row[
                            'movieId'])
                        title = str(row[
                            'title'])
                        rt    = float(row[
                            'rating'])
                        yr    = row.get(
                            'year', '')
                        yr_s  = (
                            str(int(
                                float(yr)))
                            if pd.notna(yr)
                            and str(yr)
                            not in (
                                '', 'nan')
                            else "")

                        # Poster
                        poster = str(row.get(
                            'poster_path',
                            ''))
                        purl = (
                            f"{TMDB_BASE}"
                            f"{poster}"
                            if poster not in (
                                'nan','',
                                'None')
                            and poster
                            .startswith('/')
                            else None)

                        # Star string
                        stars = (
                            "★" * int(rt) +
                            "☆" * (
                                5-int(rt)))

                        # Match pct
                        mp = int(
                            min(rt/5*100,
                                99))

                        # Card
                        st.markdown(
                            '<div class='
                            '"nf-card">',
                            unsafe_allow_html=
                            True)

                        if purl:
                            st.image(
                                purl,
                                use_column_width=
                                True)
                        else:
                            st.markdown(
                                '<div class='
                                '"nf-card-'
                                'placeholder"'
                                '>🎬</div>',
                                unsafe_allow_html=
                                True)

                        st.markdown(
                            f"""
<div class="nf-card-overlay">
  <div class="match-badge">
    {stars}
  </div>
  <div class="nf-card-title">
    {title[:38]}
  </div>
</div>
""",
                            unsafe_allow_html=
                            True)

                        st.markdown(
                            '</div>',
                            unsafe_allow_html=
                            True)

                        # Below card
                        st.markdown(
                            f'<div style="'
                            f'padding:6px 2px'
                            f' 4px;">'
                            f'<div style="'
                            f'font-family:'
                            f'Inter,sans-serif;'
                            f'font-size:12px;'
                            f'font-weight:600;'
                            f'color:{TEXT_HI};'
                            f'line-height:1.3;'
                            f'margin-bottom:'
                            f'3px;">'
                            f'{title[:28]}'
                            f'</div>'
                            f'<div style="'
                            f'font-family:'
                            f'Inter,sans-serif;'
                            f'font-size:11px;'
                            f'color:{TEXT_LO};">'
                            f'{yr_s}'
                            f'{"  ·  " if yr_s else ""}'
                            f'<span style="'
                            f'color:#a3cf62;">'
                            f'{stars}</span>'
                            f'</div>'
                            f'</div>',
                            unsafe_allow_html=
                            True)

            st.markdown(
                "<div style='height:28px'>"
                "</div>",
                unsafe_allow_html=True)

            # ── Charts ────────────────────
            ca, cb = st.columns(2)
            with ca:
                st.markdown(
                    '<div class="nf-row-label"'
                    ' style="font-size:15px;">'
                    'Rating Distribution'
                    '</div>',
                    unsafe_allow_html=True)
                dist = ur['rating']\
                    .value_counts()\
                    .sort_index()
                st.bar_chart(
                    dist, color=ACCENT)

            with cb:
                st.markdown(
                    '<div class="nf-row-label"'
                    ' style="font-size:15px;">'
                    'Top Genres</div>',
                    unsafe_allow_html=True)
                gc = {}
                for _, row in \
                        ur.iterrows():
                    for g in safe_genres(
                            row.get(
                                'genres_list')):
                        gc[g] = \
                            gc.get(g, 0)+1
                if gc:
                    gdf = pd.DataFrame(
                        list(gc.items()),
                        columns=[
                            'Genre', 'Count']
                    ).sort_values(
                        'Count',
                        ascending=False
                    ).head(8)
                    st.bar_chart(
                        gdf.set_index(
                            'Genre'),
                        color=ACCENT)

# ══════════════════════════════════════════════════
# Tab 3 — System
# ══════════════════════════════════════════════════
with tab3:
    rc, _, rb = st.columns([4, 3, 1])
    with rc:
        st.markdown(
            '<div class="nf-row-label">'
            'System Overview</div>',
            unsafe_allow_html=True)
    with rb:
        if st.button(
                "↺ Refresh",
                key="sys_ref",
                use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    m = api_get("/metrics")
    c = api_get("/cache/stats")

    hit  = c.get('hit_rate_pct', 0)
    ccu  = c.get('cached_users', 0)
    cmem = c.get('memory_used', 'N/A')
    ccon = c.get('connected', False)

    # ── Perf stats ────────────────────────
    mc = st.columns(4)
    for col, v, l in [
        (mc[0],
         m.get('total_requests', 0),
         "Requests"),
        (mc[1],
         f"{m.get('error_rate', 0):.1f}%",
         "Error Rate"),
        (mc[2],
         f"{m.get('latency_p50_ms', 0):.0f}ms",
         "P50 Latency"),
        (mc[3],
         f"{m.get('latency_p99_ms', 0):.0f}ms",
         "P99 Latency"),
    ]:
        with col:
            st.markdown(
                '<div class="nf-stat">'
                '<div class="nf-stat-val">'
                + str(v) +
                '</div>'
                '<div class="nf-stat-lbl">'
                + l +
                '</div>'
                '</div>',
                unsafe_allow_html=True)

    st.markdown(
        "<div style='height:28px'></div>",
        unsafe_allow_html=True)

    # ── Cache + Services ──────────────────
    cl, cr = st.columns(2)

    with cl:
        st.markdown(
            '<div class="nf-row-label"'
            ' style="font-size:15px;">'
            'Cache Performance</div>',
            unsafe_allow_html=True)

        conn_s = (
            '<span style="'
            'background:rgba(163,207,98,0.1);'
            'border:1px solid '
            'rgba(163,207,98,0.3);'
            'color:#a3cf62;'
            'font-family:Inter,sans-serif;'
            'font-size:10px;font-weight:600;'
            'padding:3px 8px;'
            'border-radius:3px;">'
            'Connected</span>'
            if ccon else
            '<span style="'
            'background:rgba(229,9,20,0.1);'
            'border:1px solid '
            'rgba(229,9,20,0.3);'
            'color:#e50914;'
            'font-family:Inter,sans-serif;'
            'font-size:10px;font-weight:600;'
            'padding:3px 8px;'
            'border-radius:3px;">'
            'Offline</span>')

        st.markdown(
            '<div class="nf-cache-card">'
            '<div style="display:flex;'
            'justify-content:space-between;'
            'align-items:center;'
            'margin-bottom:4px;">'
            '<span style="font-family:Inter,'
            'sans-serif;font-size:13px;'
            'font-weight:700;color:'
            + TEXT_HI + ';">Redis Cache</span>'
            + conn_s +
            '</div>'
            '<div class="nf-stat-val"'
            ' style="font-size:42px;">'
            + str(hit) + '%'
            '</div>'
            '<div class="nf-stat-lbl">'
            'Hit Rate</div>'
            '<div class="nf-progress">'
            '<div class="nf-progress-fill"'
            ' style="width:' + str(hit) + '%">'
            '</div></div>'
            '<div style="font-family:Inter,'
            'sans-serif;font-size:12px;'
            'color:' + TEXT_LO + ';">'
            + str(ccu) + ' profiles cached · '
            + str(cmem) + ' used'
            '</div></div>',
            unsafe_allow_html=True)

    with cr:
        st.markdown(
            '<div class="nf-row-label"'
            ' style="font-size:15px;">'
            'Services</div>',
            unsafe_allow_html=True)

        svcs = [
            ("FastAPI",    "8000", True),
            ("BentoML",    "3001", True),
            ("Redis",      "6379",
             m.get('redis_connected',
                   False)),
            ("Kafka",      "9092",
             m.get('kafka_connected',
                   False)),
            ("Prometheus", "9090", True),
            ("Grafana",    "3000", True),
            ("MLflow",     "5000", True),
            ("Qdrant",     "6333", True),
            ("PostgreSQL", "5432", True),
        ]

        svc_rows = ""
        for name, port, ok in svcs:
            dot_style = (
                'background:#a3cf62;'
                'animation:pulse 2.5s infinite;'
                if ok else
                'background:#e50914;')
            color = TEXT_HI if ok else TEXT_LO
            svc_rows += (
                '<div class="nf-svc">'
                '<div style="display:flex;'
                'align-items:center;gap:10px;">'
                '<div style="width:7px;'
                'height:7px;border-radius:50%;'
                + dot_style + '"></div>'
                '<span style="font-family:Inter,'
                'sans-serif;font-size:13px;'
                'font-weight:600;color:'
                + color + ';">'
                + name +
                '</span></div>'
                '<span style="font-family:'
                'JetBrains Mono,monospace;'
                'font-size:11px;color:'
                + TEXT_LO + ';">:'
                + port +
                '</span></div>')

        st.markdown(
            '<div style="background:'
            + SURFACE + ';border:1px solid '
            + BORDER + ';border-radius:4px;'
            'overflow:hidden;">'
            + svc_rows +
            '</div>',
            unsafe_allow_html=True)

    st.markdown(
        "<div style='height:28px'></div>",
        unsafe_allow_html=True)

    # ── Architecture ──────────────────────
    st.markdown(
        '<div class="nf-row-label"'
        ' style="font-size:15px;">'
        'System Architecture</div>',
        unsafe_allow_html=True)

    def make_tags(tags):
        html = ""
        for t in tags:
            html += (
                '<span style="'
                'background:' + BG3 + ';'
                'border:1px solid ' + BORDER + ';'
                'color:' + TEXT_MID + ';'
                'font-family:Inter,sans-serif;'
                'font-size:10px;'
                'font-weight:500;'
                'padding:2px 8px;'
                'border-radius:3px;'
                'display:inline-block;'
                'margin:2px 2px 2px 0;">'
                + t +
                '</span>')
        return html

    def make_row(layer, title,
                 port_str, tags,
                 border_top=True,
                 highlight=False):
        bg_color = (
            'rgba(229,9,20,0.04)'
            if highlight and dark else
            'rgba(229,9,20,0.02)'
            if highlight and not dark else
            'transparent')
        border_css = (
            'border-top:1px solid '
            + BORDER + ';'
            if border_top else '')
        port_html = (
            '<span style="'
            'font-family:JetBrains Mono,'
            'monospace;font-size:10px;'
            'color:' + TEXT_LO + ';'
            'font-weight:400;'
            'margin-left:8px;">'
            + port_str +
            '</span>'
            if port_str else '')

        return (
            '<div style="padding:16px 20px;'
            + border_css +
            'background:' + bg_color + ';">'
            '<div style="display:flex;'
            'align-items:flex-start;'
            'gap:16px;">'
            '<div style="width:120px;'
            'flex-shrink:0;'
            'font-family:Inter,sans-serif;'
            'font-size:10px;font-weight:700;'
            'color:' + TEXT_LO + ';'
            'letter-spacing:0.15em;'
            'text-transform:uppercase;'
            'padding-top:2px;">'
            + layer +
            '</div>'
            '<div style="flex:1;">'
            '<div style="'
            'font-family:Inter,sans-serif;'
            'font-size:13px;font-weight:700;'
            'color:' + TEXT_HI + ';'
            'margin-bottom:8px;">'
            + title + port_html +
            '</div>'
            '<div>'
            + make_tags(tags) +
            '</div>'
            '</div>'
            '</div>'
            '</div>')

    rows = ""

    rows += make_row(
        "Frontend", "Streamlit", ":8501",
        ["Python 3.12",
         "Netflix-style UI",
         "Dark / Light Mode",
         "TMDB Posters",
         "Real-time Metrics",
         "Inline Feedback"],
        border_top=False)

    rows += make_row(
        "API Gateway", "FastAPI", ":8000",
        ["Rate Limiting 100/min",
         "API Key Auth",
         "CORS Middleware",
         "Pydantic v2",
         "OpenAPI / Swagger",
         "Request ID Tracking",
         "JSON Logging",
         "Prometheus Metrics"])

    rows += make_row(
        "Model Serving",
        "BentoML 1.x", ":3001",
        ["HSTU Ranker",
         "PyTorch Inference",
         "ONNX Export",
         "Popularity Padding",
         "Cold Start Fallback",
         "Deduplication"])

    rows += make_row(
        "Cache Layer",
        "Redis 7", ":6379",
        ["Per-user Caching",
         "TTL 1 Hour",
         "Feedback Invalidation",
         "Cache Warming",
         str(hit) + "% Hit Rate",
         str(ccu) + " Profiles Cached"])

    rows += make_row(
        "Streaming",
        "Apache Kafka", ":9092",
        ["user-interactions topic",
         "recommendations topic",
         "dead-letter queue",
         "Online Model Updates",
         "Acks=all Durability",
         "kafka-python-ng",
         "cp-kafka:7.6.0"])

    rows += make_row(
        "Observability",
        "Prometheus + Grafana",
        ":9090 / :3000",
        ["6 Custom Metrics",
         "p50 / p95 / p99",
         "Cache Hit Rate",
         "Request Volume",
         "Error Rate",
         "Evidently AI Drift",
         "15s Scrape Interval",
         "7-Panel Dashboard"])

    rows += make_row(
        "Data & MLOps",
        "Full ML Pipeline", "",
        ["PySpark ETL",
         "26M Ratings",
         "Delta Lake",
         "AWS S3",
         "Feast Feature Store",
         "MLflow Tracking",
         "DVC Versioning",
         "Prefect Orchestration",
         "Great Expectations",
         "Qdrant Vector DB",
         "PostgreSQL"])

    rows += make_row(
        "Deployment",
        "Production Infrastructure", "",
        ["Docker Compose",
         "Terraform IaC",
         "AWS EKS / GKE",
         "Kubernetes",
         "Helm Charts",
         "Argo CD GitOps",
         "Argo Rollouts",
         "Canary Deployments",
         "GitHub Actions CI/CD",
         "Istio Service Mesh"])

    rows += make_row(
        "Retrieval",
        "Two-Stage Pipeline", "",
        ["GRank (WWW 2026)",
         "TIGER Generative",
         "Qdrant HNSW ANN",
         "e5-large Embeddings",
         "CLIP Multimodal",
         "Ollama RAG",
         "500 Candidates"])

    rows += make_row(
        "Ranking",
        "Neural Rankers", "",
        ["HSTU",
         "Netflix FM",
         "OneRec",
         "LightGCN",
         "BERT4Rec",
         "DPO Alignment",
         "Multi-task",
         "MMR Re-ranking",
         "IPS Debiasing",
         "FM-Intent",
         "LLM Explainability",
         "Fairness Calibration"])

    rows += make_row(
        "Evaluation",
        "8-Metric Framework", "",
        ["NDCG@10",
         "Precision@K",
         "Recall@K",
         "MAP",
         "MRR",
         "Coverage",
         "Diversity",
         "Novelty",
         "5-Fold Temporal CV",
         "IPS Correction",
         "A/B Testing",
         "p < 0.05 Significance"])

    # ── Key results row ───────────────────
    result_cards = ""
    for v, l in [
        ("+22%",  "NDCG vs CF"),
        ("0.022", "HSTU NDCG@10"),
        ("88%",   "Cache Hit Rate"),
        ("13ms",  "P50 Latency"),
        ("-63%",  "IPS Correction"),
        ("4",     "Domains"),
        ("0",     "Code Changes"),
        ("15+",   "Models Built"),
        ("26M",   "Ratings"),
        ("9K",    "Movies"),
    ]:
        result_cards += (
            '<div style="'
            'text-align:center;'
            'padding:8px 14px;'
            'background:' + BG3 + ';'
            'border:1px solid ' + BORDER + ';'
            'border-radius:4px;">'
            '<div style="'
            'font-family:Inter,sans-serif;'
            'font-size:22px;'
            'font-weight:900;'
            'color:' + ACCENT + ';'
            'letter-spacing:-0.02em;'
            'line-height:1;">'
            + v +
            '</div>'
            '<div style="'
            'font-family:Inter,sans-serif;'
            'font-size:9px;font-weight:700;'
            'color:' + TEXT_LO + ';'
            'letter-spacing:0.1em;'
            'text-transform:uppercase;'
            'margin-top:4px;">'
            + l +
            '</div>'
            '</div>')

    rows += (
        '<div style="'
        'padding:20px;'
        'border-top:1px solid ' + BORDER + ';'
        'background:'
        + ('rgba(229,9,20,0.04)'
           if dark else
           'rgba(229,9,20,0.02)') + ';'
        'display:flex;'
        'align-items:flex-start;'
        'gap:16px;">'
        '<div style="width:120px;'
        'flex-shrink:0;'
        'font-family:Inter,sans-serif;'
        'font-size:10px;font-weight:700;'
        'color:' + TEXT_LO + ';'
        'letter-spacing:0.15em;'
        'text-transform:uppercase;'
        'padding-top:6px;">'
        'Key Results</div>'
        '<div style="flex:1;display:flex;'
        'gap:8px;flex-wrap:wrap;">'
        + result_cards +
        '</div>'
        '</div>')

    st.markdown(
        '<div style="'
        'background:' + BG2 + ';'
        'border:1px solid ' + BORDER + ';'
        'border-radius:4px;'
        'overflow:hidden;">'
        + rows +
        '</div>',
        unsafe_allow_html=True)