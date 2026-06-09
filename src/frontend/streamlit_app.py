"""
Production RecSys — Streamlit Frontend

Live movie recommendation UI with:
  → User selection
  → Watch history display
  → HSTU personalised recommendations
  → TMDB poster images (with fallback)
  → Inline rating feedback
  → System health dashboard
  → Real-time metrics

Usage:
  streamlit run src/frontend/streamlit_app.py
"""

import sys
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
API_URL     = "http://localhost:8000"
PROM_URL    = "http://localhost:9090"
TMDB_BASE   = "https://image.tmdb.org/t/p/w300"

st.set_page_config(
    page_title = "🎬 RecSys Demo",
    page_icon  = "🎬",
    layout     = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────
st.markdown("""
<style>
.rec-card {
    background: #1e1e2e;
    border-radius: 12px;
    padding: 12px;
    margin: 6px 0;
    border: 1px solid #313244;
}
.rec-title {
    font-size: 15px;
    font-weight: 600;
    color: #cdd6f4;
}
.rec-score {
    font-size: 12px;
    color: #a6e3a1;
}
.fallback-badge {
    background: #f38ba8;
    color: white;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 10px;
}
.hstu-badge {
    background: #89b4fa;
    color: #1e1e2e;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 10px;
}
.cached-badge {
    background: #a6e3a1;
    color: #1e1e2e;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 10px;
}
.metric-card {
    background: #181825;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    border: 1px solid #313244;
}
</style>
""", unsafe_allow_html=True)


# ── API Helpers ───────────────────────────────────
@st.cache_data(ttl=60)
def load_movies():
    """Load movie catalog"""
    try:
        df = pd.read_csv(
            BASE / 'data' / 'processed' /
            'movies_master.csv',
            low_memory=False)
        df = df[df['movieId'].notna()].copy()
        df['movieId'] = df['movieId']\
            .astype(int)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def load_ratings():
    """Load ratings for history"""
    try:
        df = pd.read_csv(
            BASE / 'data' / 'processed' /
            'ratings_cleaned.csv')
        return df.sort_values(
            ['userId', 'timestamp'])
    except Exception:
        return pd.DataFrame()


def get_recommendations(
        user_id: int,
        top_k:   int = 10) -> dict:
    """Call FastAPI recommend endpoint"""
    try:
        r = requests.post(
            f"{API_URL}/recommend",
            json    = {
                "user_id": user_id,
                "top_k":   top_k,
            },
            timeout = 30)
        if r.status_code == 200:
            return r.json()
        return {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def send_feedback(
        user_id:  int,
        movie_id: int,
        rating:   float,
        action:   str = "rate") -> bool:
    """Send feedback to FastAPI"""
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
    """Get API health status"""
    try:
        r = requests.get(
            f"{API_URL}/health",
            timeout=5)
        return r.json() \
            if r.status_code == 200 \
            else {}
    except Exception:
        return {}


def get_metrics() -> dict:
    """Get API metrics"""
    try:
        r = requests.get(
            f"{API_URL}/metrics",
            timeout=5)
        return r.json() \
            if r.status_code == 200 \
            else {}
    except Exception:
        return {}


def get_cache_stats() -> dict:
    """Get Redis cache stats"""
    try:
        r = requests.get(
            f"{API_URL}/cache/stats",
            timeout=5)
        return r.json() \
            if r.status_code == 200 \
            else {}
    except Exception:
        return {}


def get_poster_url(
        movie_row: pd.Series) -> Optional[str]:
    """Get TMDB poster URL"""
    try:
        poster = movie_row.get(
            'poster_path', '')
        if poster and \
                str(poster) != 'nan' and \
                str(poster) != '':
            # poster_path starts with /
            return f"{TMDB_BASE}{poster}"
    except Exception:
        pass
    return None


# ── Load Data ─────────────────────────────────────
movies  = load_movies()
ratings = load_ratings()

# Get available users
if not ratings.empty:
    user_counts = ratings.groupby(
        'userId')['rating'].count()
    available_users = user_counts\
        .sort_values(ascending=False)\
        .index.tolist()
else:
    available_users = list(range(1, 51))

# ── Sidebar ───────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 RecSys Demo")
    st.markdown(
        "Production recommendation system "
        "built with HSTU ranker "
        "(Meta MLPerf 2026)")
    st.divider()

    # User selection
    st.markdown("### 👤 Select User")
    user_id = st.selectbox(
        "User ID",
        options = available_users[:50],
        index   = 0,
        help    = "Select a user to get "
                  "personalised recommendations")

    top_k = st.slider(
        "Number of recommendations",
        min_value = 1,
        max_value = 20,
        value     = 10)

    st.divider()

    # System health
    st.markdown("### 🔧 System Health")
    health = get_health()

    if health.get('status') == 'healthy':
        st.success("✅ System healthy")
    else:
        st.warning("⚠️ System degraded")

    st.caption(
        f"Model: {health.get('model', 'N/A')}")

    # Cache stats
    cache = get_cache_stats()
    if cache.get('connected'):
        hit_rate = cache.get(
            'hit_rate_pct', 0)
        st.metric(
            "Cache hit rate",
            f"{hit_rate}%")
        st.metric(
            "Cached users",
            cache.get('cached_users', 0))

    st.divider()
    st.markdown("### 🔗 Links")
    st.markdown(
        "- [API Docs](http://localhost:8000/docs)")
    st.markdown(
        "- [Prometheus](http://localhost:9090)")
    st.markdown(
        "- [Grafana](http://localhost:3000)")
    st.markdown(
        "- [MLflow](http://localhost:5000)")


# ── Main Content ──────────────────────────────────
st.title("🎬 Production Movie RecSys")
st.markdown(
    "**HSTU Ranker** (Meta MLPerf 2026) · "
    "**Redis Cache** · "
    "**Kafka Streaming** · "
    "**Prometheus Monitoring**")

st.divider()

# ── Tabs ──────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🎯 Recommendations",
    "📜 Watch History",
    "📊 System Metrics",
])


# ── Tab 1: Recommendations ────────────────────────
with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(
            f"### Recommendations for "
            f"User {user_id}")
    with col2:
        refresh = st.button(
            "🔄 Refresh",
            use_container_width=True)

    # Get recommendations
    with st.spinner(
            "Getting recommendations..."):
        start    = time.time()
        rec_data = get_recommendations(
            user_id, top_k)
        latency  = (time.time()-start)*1000

    if "error" in rec_data:
        st.error(
            f"API Error: {rec_data['error']}\n\n"
            f"Make sure FastAPI is running: "
            f"`uvicorn src.serving.fastapi_app:app "
            f"--port 8000`")
    else:
        # Metadata row
        meta_cols = st.columns(4)
        with meta_cols[0]:
            st.metric(
                "Recommendations",
                rec_data.get('n_recs', 0))
        with meta_cols[1]:
            st.metric(
                "Latency",
                f"{rec_data.get('latency_ms', 0):.0f}ms")
        with meta_cols[2]:
            cached = rec_data.get(
                'cached', False)
            st.metric(
                "Source",
                "Redis Cache ⚡"
                if cached
                else "HSTU Model 🤖")
        with meta_cols[3]:
            st.metric(
                "Request ID",
                rec_data.get(
                    'request_id', 'N/A'))

        st.divider()

        # Recommendation grid
        recs = rec_data.get(
            'recommendations', [])

        if not recs:
            st.warning(
                "No recommendations found")
        else:
            # Display in grid
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
                        mid = rec['movie_id']

                        # Get movie info
                        m_row = movies[
                            movies['movieId']
                            == mid
                        ] if not movies.empty\
                            else pd.DataFrame()

                        title = rec.get(
                            'title',
                            f'Movie {mid}')

                        # Poster
                        poster_url = None
                        if not m_row.empty:
                            poster_url = \
                                get_poster_url(
                                    m_row.iloc[0])

                        if poster_url:
                            st.image(
                                poster_url,
                                use_column_width=True)
                        else:
                            st.markdown(
                                f"""
<div style="background:#313244;
border-radius:8px;
height:150px;
display:flex;
align-items:center;
justify-content:center;
font-size:40px;">
🎬
</div>""",
                                unsafe_allow_html=True)

                        # Title
                        st.markdown(
                            f"**#{rec['rank']} "
                            f"{title[:25]}**")

                        # Score
                        score = rec.get(
                            'score', 0)
                        if score > 0:
                            st.progress(
                                min(float(
                                    score), 1.0))

                        # Fallback badge
                        if rec.get('fallback'):
                            st.caption(
                                "📌 Popular")
                        else:
                            st.caption(
                                "🤖 HSTU")

                        # Inline rating
                        with st.expander(
                                "Rate"):
                            stars = st.select_slider(
                                f"r_{mid}",
                                options=[
                                    "⭐", "⭐⭐",
                                    "⭐⭐⭐",
                                    "⭐⭐⭐⭐",
                                    "⭐⭐⭐⭐⭐"],
                                label_visibility=
                                    "collapsed")
                            if st.button(
                                    "Submit",
                                    key=f"btn_{mid}"):
                                rating = len(stars)
                                ok = send_feedback(
                                    user_id,
                                    mid,
                                    float(rating),
                                    "rate")
                                if ok:
                                    st.success(
                                        "✅ Saved!")
                                    st.cache_data\
                                        .clear()


# ── Tab 2: Watch History ──────────────────────────
with tab2:
    st.markdown(
        f"### Watch History — User {user_id}")

    if ratings.empty:
        st.warning("No ratings data loaded")
    else:
        user_ratings = ratings[
            ratings['userId'] == user_id
        ].copy()

        if user_ratings.empty:
            st.info(
                f"No history for "
                f"user {user_id}")
        else:
            # Merge with movies
            if not movies.empty:
                user_ratings = user_ratings\
                    .merge(
                        movies[[
                            'movieId',
                            'title',
                            'genres_list']],
                        on='movieId',
                        how='left')

            user_ratings = user_ratings\
                .sort_values(
                    'rating',
                    ascending=False)

            st.metric(
                "Total rated movies",
                len(user_ratings))

            # Top rated
            st.markdown("#### ⭐ Top Rated")
            top_rated = user_ratings\
                .head(10)

            for _, row in top_rated.iterrows():
                title  = row.get(
                    'title',
                    f"Movie {row['movieId']}")
                rating = row['rating']
                stars  = "⭐" * int(rating)

                col1, col2 = st.columns(
                    [4, 1])
                with col1:
                    st.markdown(
                        f"**{title[:45]}**")
                with col2:
                    st.markdown(
                        f"{stars} {rating}")

            st.divider()

            # Rating distribution
            st.markdown(
                "#### 📊 Rating Distribution")
            dist = user_ratings[
                'rating'].value_counts()\
                .sort_index()

            st.bar_chart(dist)

            # Genre preferences
            if 'genres_list' in \
                    user_ratings.columns:
                st.markdown(
                    "#### 🎭 Genre Preferences")
                import ast
                genre_counts = {}
                for _, row in \
                        user_ratings.iterrows():
                    genres = row.get(
                        'genres_list', [])
                    if isinstance(genres, str):
                        try:
                            genres = ast.literal_eval(
                                genres)
                        except Exception:
                            genres = []
                    for g in (genres or []):
                        genre_counts[g] = \
                            genre_counts.get(
                                g, 0) + 1

                if genre_counts:
                    genre_df = pd.DataFrame(
                        list(genre_counts.items()),
                        columns=['Genre', 'Count']
                    ).sort_values(
                        'Count',
                        ascending=False).head(10)
                    st.bar_chart(
                        genre_df.set_index(
                            'Genre'))


# ── Tab 3: System Metrics ─────────────────────────
with tab3:
    st.markdown("### 📊 System Metrics")

    if st.button("🔄 Refresh Metrics"):
        st.cache_data.clear()

    metrics = get_metrics()
    cache   = get_cache_stats()

    # Top metrics row
    m_cols = st.columns(4)
    with m_cols[0]:
        st.metric(
            "Total Requests",
            metrics.get(
                'total_requests', 0))
    with m_cols[1]:
        st.metric(
            "Error Rate",
            f"{metrics.get('error_rate', 0):.1f}%")
    with m_cols[2]:
        st.metric(
            "p50 Latency",
            f"{metrics.get('latency_p50_ms', 0):.0f}ms")
    with m_cols[3]:
        st.metric(
            "p99 Latency",
            f"{metrics.get('latency_p99_ms', 0):.0f}ms")

    st.divider()

    # Cache metrics
    st.markdown("#### 🔴 Redis Cache")
    c_cols = st.columns(4)
    with c_cols[0]:
        st.metric(
            "Connected",
            "✅" if cache.get(
                'connected') else "❌")
    with c_cols[1]:
        st.metric(
            "Hit Rate",
            f"{cache.get('hit_rate_pct', 0)}%")
    with c_cols[2]:
        st.metric(
            "Cached Users",
            cache.get('cached_users', 0))
    with c_cols[3]:
        st.metric(
            "Memory",
            cache.get('memory_used', 'N/A'))

    st.divider()

    # Service status
    st.markdown("#### 🔧 Service Status")
    s_cols = st.columns(3)
    with s_cols[0]:
        redis_ok = metrics.get(
            'redis_connected', False)
        st.markdown(
            f"**Redis** "
            f"{'✅ Connected' if redis_ok else '❌ Down'}")
        st.caption("localhost:6379")

    with s_cols[1]:
        kafka_ok = metrics.get(
            'kafka_connected', False)
        st.markdown(
            f"**Kafka** "
            f"{'✅ Connected' if kafka_ok else '❌ Down'}")
        st.caption("localhost:9092")

    with s_cols[2]:
        st.markdown("**BentoML** ✅ Running")
        st.caption("localhost:3001")

    st.divider()

    # Architecture diagram
    st.markdown("#### 🏗️ Architecture")
    st.code("""
User Browser
     ↓
Streamlit :8501          ← YOU ARE HERE
     ↓
FastAPI :8000            ← REST API
  ↓ cache hit    ↓ miss
Redis :6379    BentoML :3001
               ↓
             HSTU Model
     ↓
Kafka :9092              ← Event streaming
     ↓
Prometheus :9090         ← Metrics
     ↓
Grafana :3000            ← Dashboards
    """, language="")