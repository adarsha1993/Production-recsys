"""
FastAPI Production REST API
Sits in front of BentoML service.

Features:
  → Rate limiting (100 req/min per IP)
  → API key authentication
  → Structured JSON logging
  → CORS for Streamlit UI
  → OpenAPI docs at /docs
  → Health + readiness probes
  → Request ID tracking
  → Redis caching (Day 31)
  → Kafka producer (Day 32)
  → Prometheus metrics (Day 33)

Usage:
  uvicorn src.serving.fastapi_app:app
          --host 0.0.0.0 --port 8000
          --reload
"""

import os
import sys
import time
import uuid
import logging
import numpy as np
import httpx

from fastapi import (
    FastAPI, HTTPException,
    Request, Depends,
    Header, status)
from fastapi.middleware.cors import (
    CORSMiddleware)
from fastapi.responses import (
    JSONResponse, Response)
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST)

# ── Paths ─────────────────────────────────────────
BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))

# ── Redis cache ───────────────────────────────────
from src.serving.redis_cache import (
    RecommendationCache)

cache = RecommendationCache(
    host   = 'localhost',
    port   = 6379,
    ttl    = 3600,
    prefix = 'rec')

# ── Kafka producer ────────────────────────────────
from src.serving.kafka_producer import (
    InteractionProducer)

producer = InteractionProducer(
    bootstrap_servers='localhost:9092')

# ── Prometheus Metrics ────────────────────────────
rec_requests_total = Counter(
    'rec_requests_total',
    'Total recommendation requests',
    ['model', 'cached', 'status'])

rec_latency_seconds = Histogram(
    'rec_latency_seconds',
    'Recommendation latency in seconds',
    ['endpoint'],
    buckets=[.005, .01, .025, .05,
             .1, .25, .5, 1.0, 2.5])

cache_hits_total = Counter(
    'cache_hits_total',
    'Total Redis cache hits')

cache_misses_total = Counter(
    'cache_misses_total',
    'Total Redis cache misses')

active_users_gauge = Gauge(
    'active_users_total',
    'Number of users with cached recs')

kafka_events_total = Counter(
    'kafka_events_total',
    'Total Kafka events sent',
    ['topic', 'status'])

model_info = Gauge(
    'model_info',
    'Model information',
    ['model_name', 'version'])
model_info.labels(
    model_name='HSTU',
    version='1.0.0').set(1)

# ── Logging ───────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s %(levelname)s '
              '%(name)s %(message)s')
log = logging.getLogger("fastapi_app")

# ── Rate limiter ──────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address)

# ── API Keys ──────────────────────────────────────
VALID_API_KEYS = {
    "dev-key-123":  "development",
    "prod-key-456": "production",
    "test-key-789": "testing",
}

# BentoML service URL
BENTOML_URL = os.getenv(
    "BENTOML_URL",
    "http://localhost:3001")


# ── Pydantic Models ───────────────────────────────
class RecommendRequest(BaseModel):
    user_id: int = Field(
        ...,
        description="User ID",
        example=481)
    top_k: int = Field(
        default=10,
        ge=1, le=50,
        description="Number of recommendations",
        example=10)
    include_scores: bool = Field(
        default=True,
        description="Include model scores")


class FeedbackRequest(BaseModel):
    user_id:   int   = Field(
        ..., description="User ID")
    movie_id:  int   = Field(
        ..., description="Movie ID")
    rating:    float = Field(
        ..., ge=0.5, le=5.0,
        description="Rating 0.5-5.0")
    action:    str   = Field(
        default="watch",
        description="watch/like/share")
    watch_pct: Optional[float] = Field(
        default=None,
        ge=0.0, le=1.0,
        description="Watch completion 0-1")


class MovieRec(BaseModel):
    rank:       int
    movie_id:   int
    title:      str
    score:      float
    cold_start: bool
    fallback:   bool


class RecommendResponse(BaseModel):
    request_id:      str
    user_id:         int
    recommendations: List[MovieRec]
    model:           str
    latency_ms:      float
    n_recs:          int
    cached:          bool = False


class HealthResponse(BaseModel):
    status:    str
    api:       str
    model:     str
    version:   str
    timestamp: float


# ── FastAPI App ───────────────────────────────────
app = FastAPI(
    title       = "Production RecSys API",
    description = """
## Production Recommendation System API

Built with HSTU ranker (Meta MLPerf 2026).

### Endpoints
- **POST /recommend**    — Get recommendations
- **POST /feedback**     — Log interaction
- **GET  /health**       — Health check
- **GET  /metrics**      — Request metrics
- **GET  /prometheus**   — Prometheus metrics
- **GET  /cache/stats**  — Redis stats
- **GET  /kafka/stats**  — Kafka stats

### Authentication
Include `X-API-Key` header with valid key.

### Rate Limiting
100 requests per minute per IP.
    """,
    version  = "1.0.0",
    docs_url = "/docs",
    redoc_url= "/redoc",
)

# ── Middleware ────────────────────────────────────
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Rate limit error handler ──────────────────────
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(
        request: Request,
        exc: RateLimitExceeded):
    return JSONResponse(
        status_code = 429,
        content     = {
            "error":       "Rate limit exceeded",
            "message":     "100 req/min per IP",
            "retry_after": "60 seconds",
        })


# ── Request logging middleware ────────────────────
@app.middleware("http")
async def log_requests(
        request: Request,
        call_next):
    request_id = str(uuid.uuid4())[:8]
    start      = time.time()

    request.state.request_id = request_id
    response = await call_next(request)
    latency  = (time.time()-start)*1000

    log.info(
        f"rid={request_id} "
        f"method={request.method} "
        f"path={request.url.path} "
        f"status={response.status_code} "
        f"latency={latency:.1f}ms")

    response.headers[
        "X-Request-ID"] = request_id
    response.headers[
        "X-Latency-Ms"] = str(
        round(latency, 1))

    return response


# ── Auth dependency ───────────────────────────────
async def verify_api_key(
        x_api_key: Optional[str] = Header(
            default=None)):
    if os.getenv("ENV", "dev") == "dev":
        return "development"
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header required")
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key")
    return VALID_API_KEYS[x_api_key]


# ── Request metrics ───────────────────────────────
request_metrics = {
    "total":     0,
    "success":   0,
    "errors":    0,
    "latencies": [],
}


# ── Endpoints ─────────────────────────────────────
@app.get("/",
         tags=["Operations"])
async def root():
    """API root"""
    return {
        "message": "Production RecSys API",
        "docs":    "/docs",
        "health":  "/health",
        "version": "1.0.0",
    }


@app.get("/health",
         response_model=HealthResponse,
         tags=["Operations"])
async def health():
    """Kubernetes liveness probe."""
    bentoml_status = "unknown"
    try:
        async with httpx.AsyncClient() \
                as client:
            r = await client.post(
                f"{BENTOML_URL}/health",
                json    = {"request":
                           {"ping": "ping"}},
                timeout = 2.0)
            bentoml_status = "healthy" \
                if r.status_code == 200 \
                else "degraded"
    except Exception:
        bentoml_status = "unreachable"

    redis_ok = cache.connected
    kafka_ok = producer.connected

    return HealthResponse(
        status    = "healthy"
                    if bentoml_status ==
                    "healthy" else "degraded",
        api       = "FastAPI",
        model     = (
            f"HSTU via BentoML"
            f"({bentoml_status}) "
            f"Redis({'ok' if redis_ok else 'down'}) "
            f"Kafka({'ok' if kafka_ok else 'down'})"),
        version   = "1.0.0",
        timestamp = time.time(),
    )


@app.get("/readiness",
         tags=["Operations"])
async def readiness():
    """Kubernetes readiness probe."""
    return {
        "status":    "ready",
        "timestamp": time.time()}


@app.get("/prometheus",
         tags=["Operations"])
async def prometheus_metrics():
    """
    Prometheus metrics endpoint.
    Scraped every 15s by Prometheus.
    """
    return Response(
        content    = generate_latest(),
        media_type = CONTENT_TYPE_LATEST)


@app.post("/recommend",
          response_model=RecommendResponse,
          tags=["Recommendations"])
@limiter.limit("100/minute")
async def recommend(
        request:     Request,
        body:        RecommendRequest,
        environment: str = Depends(
            verify_api_key)):
    """
    Get personalised movie recommendations.
    Results cached in Redis for 1 hour.
    """
    start      = time.time()
    request_id = getattr(
        request.state, 'request_id',
        str(uuid.uuid4())[:8])

    request_metrics['total'] += 1

    try:
        # ── Check Redis cache ─────────────
        cached_recs = cache.get(body.user_id)

        if cached_recs is not None:
            latency = time.time() - start
            request_metrics['success'] += 1
            request_metrics['latencies']\
                .append(latency * 1000)

            # Prometheus
            cache_hits_total.inc()
            rec_requests_total.labels(
                model  = 'HSTU+Cache',
                cached = 'true',
                status = 'success').inc()
            rec_latency_seconds.labels(
                endpoint='/recommend'
            ).observe(latency)

            log.info(
                f"rid={request_id} "
                f"user={body.user_id} "
                f"CACHE HIT "
                f"latency={latency*1000:.1f}ms")

            if not body.include_scores:
                for r in cached_recs:
                    r['score'] = 0.0

            producer.send_recommendation_logged(
                user_id    = body.user_id,
                recs       = cached_recs,
                model      = 'HSTU+Cache',
                latency_ms = latency * 1000,
                cached     = True)

            return RecommendResponse(
                request_id      = request_id,
                user_id         = body.user_id,
                recommendations = cached_recs,
                model           = 'HSTU+Cache',
                latency_ms      = round(
                    latency * 1000, 2),
                n_recs          = len(
                    cached_recs),
                cached          = True,
            )

        # ── Cache miss → BentoML ──────────
        async with httpx.AsyncClient() \
                as client:
            r = await client.post(
                f"{BENTOML_URL}/recommend",
                json    = {
                    "request": {
                        "user_id": body.user_id,
                        "top_k":   body.top_k,
                    }
                },
                timeout = 30.0)

        if r.status_code != 200:
            raise HTTPException(
                status_code = 502,
                detail      = "Model service "
                              "unavailable")

        data = r.json()
        recs = data.get(
            "recommendations", [])

        cache.set(body.user_id, recs)

        if not body.include_scores:
            for rec in recs:
                rec['score'] = 0.0

        latency = time.time() - start
        request_metrics['success'] += 1
        request_metrics['latencies']\
            .append(latency * 1000)

        # Prometheus
        cache_misses_total.inc()
        rec_requests_total.labels(
            model  = 'HSTU',
            cached = 'false',
            status = 'success').inc()
        rec_latency_seconds.labels(
            endpoint='/recommend'
        ).observe(latency)

        log.info(
            f"rid={request_id} "
            f"user={body.user_id} "
            f"CACHE MISS "
            f"n_recs={len(recs)} "
            f"latency={latency*1000:.1f}ms")

        producer.send_recommendation_logged(
            user_id    = body.user_id,
            recs       = recs,
            model      = 'HSTU',
            latency_ms = latency * 1000,
            cached     = False)

        return RecommendResponse(
            request_id      = request_id,
            user_id         = body.user_id,
            recommendations = recs,
            model           = 'HSTU',
            latency_ms      = round(
                latency * 1000, 2),
            n_recs          = len(recs),
            cached          = False,
        )

    except httpx.TimeoutException:
        request_metrics['errors'] += 1
        rec_requests_total.labels(
            model  = 'HSTU',
            cached = 'false',
            status = 'timeout').inc()
        raise HTTPException(
            status_code = 504,
            detail      = "Model timeout")

    except Exception as e:
        request_metrics['errors'] += 1
        rec_requests_total.labels(
            model  = 'HSTU',
            cached = 'false',
            status = 'error').inc()
        log.error(
            f"rid={request_id} "
            f"error={str(e)}")
        raise HTTPException(
            status_code = 500,
            detail      = str(e))


@app.post("/feedback",
          tags=["Feedback"])
@limiter.limit("200/minute")
async def feedback(
        request:     Request,
        body:        FeedbackRequest,
        environment: str = Depends(
            verify_api_key)):
    """
    Log user interaction.
    Sends to Kafka + invalidates cache.
    """
    request_id = getattr(
        request.state, 'request_id',
        str(uuid.uuid4())[:8])

    log.info(
        f"rid={request_id} "
        f"feedback user={body.user_id} "
        f"movie={body.movie_id} "
        f"rating={body.rating} "
        f"action={body.action}")

    # 1. Invalidate Redis cache
    invalidated = cache.invalidate(
        body.user_id)

    # 2. Send to Kafka + track metric
    kafka_sent = producer.send_interaction(
        user_id   = body.user_id,
        movie_id  = body.movie_id,
        rating    = body.rating,
        action    = body.action,
        watch_pct = body.watch_pct,
    )
    kafka_events_total.labels(
        topic  = 'user-interactions',
        status = 'sent' if kafka_sent
                 else 'logged').inc()

    # 3. Forward to BentoML non-blocking
    try:
        async with httpx.AsyncClient() \
                as client:
            await client.post(
                f"{BENTOML_URL}/feedback",
                json    = {
                    "request": {
                        "user_id":  body.user_id,
                        "movie_id": body.movie_id,
                        "rating":   body.rating,
                        "action":   body.action,
                    }
                },
                timeout = 5.0)
    except Exception:
        pass

    return {
        "status":            "logged",
        "request_id":        request_id,
        "user_id":           body.user_id,
        "movie_id":          body.movie_id,
        "timestamp":         time.time(),
        "cache_invalidated": invalidated,
        "kafka_sent":        kafka_sent,
        "message":           "Queued for "
                             "Kafka pipeline",
    }


@app.get("/metrics",
         tags=["Operations"])
async def metrics():
    """Request metrics (JSON format)."""
    lats = request_metrics['latencies']
    return {
        "total_requests":  request_metrics[
            'total'],
        "successful":      request_metrics[
            'success'],
        "errors":          request_metrics[
            'errors'],
        "error_rate":      round(
            request_metrics['errors'] /
            max(request_metrics['total'],
                1) * 100, 2),
        "latency_p50_ms":  round(float(
            np.percentile(lats, 50)),
            1) if lats else 0,
        "latency_p95_ms":  round(float(
            np.percentile(lats, 95)),
            1) if lats else 0,
        "latency_p99_ms":  round(float(
            np.percentile(lats, 99)),
            1) if lats else 0,
        "kafka_connected": producer.connected,
        "redis_connected": cache.connected,
        "timestamp":       time.time(),
    }


@app.get("/cache/stats",
         tags=["Operations"])
async def cache_stats():
    """Redis cache statistics."""
    return cache.stats()


@app.post("/cache/invalidate/{user_id}",
          tags=["Operations"])
async def cache_invalidate(user_id: int):
    """Manually invalidate user cache."""
    invalidated = cache.invalidate(user_id)
    return {
        "user_id":     user_id,
        "invalidated": invalidated,
        "timestamp":   time.time(),
    }


@app.get("/kafka/stats",
         tags=["Operations"])
async def kafka_stats():
    """Kafka producer statistics."""
    return {
        "connected":          producer.connected,
        "bootstrap_servers":  producer.bootstrap_servers,
        "topics": {
            "interactions":   "user-interactions",
            "recommendations": "recommendations",
            "dead_letter":    "dead-letter",
        },
        "timestamp": time.time(),
    }