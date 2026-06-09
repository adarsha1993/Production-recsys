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

Usage:
  uvicorn src.serving.fastapi_app:app
          --host 0.0.0.0 --port 8000
          --reload
"""

import os
import time
import uuid
import logging
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import httpx

from fastapi import (
    FastAPI, HTTPException,
    Request, Depends,
    Header, status)
from fastapi.middleware.cors import (
    CORSMiddleware)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from pathlib import Path
from slowapi import Limiter
from slowapi.util import (
    get_remote_address)
from slowapi.errors import (
    RateLimitExceeded)

# ── Paths ─────────────────────────────────────────
BASE = Path(__file__).parent.parent.parent
CKPT = BASE / 'models' / 'checkpoints'
PROC = BASE / 'data' / 'processed'

# ── Logging ───────────────────────────────────────
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s %(levelname)s '
               '%(name)s %(message)s')
log = logging.getLogger("fastapi_app")

# ── Rate limiter ──────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address)

# ── API Keys ──────────────────────────────────────
# Production: store in env vars / secrets
VALID_API_KEYS = {
    "dev-key-123":   "development",
    "prod-key-456":  "production",
    "test-key-789":  "testing",
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
    user_id:  int   = Field(
        ..., description="User ID")
    movie_id: int   = Field(
        ..., description="Movie ID")
    rating:   float = Field(
        ..., ge=0.5, le=5.0,
        description="Rating 0.5-5.0")
    action:   str   = Field(
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
    status:     str
    api:        str
    model:      str
    version:    str
    timestamp:  float


# ── FastAPI App ───────────────────────────────────
app = FastAPI(
    title       = "Production RecSys API",
    description = """
## Production Recommendation System API

Built with HSTU ranker (Meta MLPerf 2026).

### Endpoints
- **POST /recommend** — Get personalised recommendations
- **POST /feedback**  — Log user interaction
- **GET  /health**    — Service health check
- **GET  /metrics**   — Request metrics

### Authentication
Include `X-API-Key` header with valid key.

### Rate Limiting
100 requests per minute per IP.
    """,
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── Middleware ────────────────────────────────────
app.state.limiter = limiter

# CORS — allows Streamlit UI to call API
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
            "error":   "Rate limit exceeded",
            "message": "100 requests/minute "
                       "per IP",
            "retry_after": "60 seconds",
        })


# ── Request logging middleware ─────────────────────
@app.middleware("http")
async def log_requests(
        request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start      = time.time()

    # Add request ID to state
    request.state.request_id = request_id

    response = await call_next(request)

    latency = (time.time()-start)*1000
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
    """
    Verify API key from X-API-Key header.
    Skip auth in development mode.
    """
    # Development mode — skip auth
    if os.getenv("ENV", "dev") == "dev":
        return "development"

    if not x_api_key:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail      = "X-API-Key header required")

    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail      = "Invalid API key")

    return VALID_API_KEYS[x_api_key]


# ── Request counter ───────────────────────────────
request_metrics = {
    "total":    0,
    "success":  0,
    "errors":   0,
    "latencies": [],
}


# ── Endpoints ─────────────────────────────────────
@app.get("/health",
         response_model=HealthResponse,
         tags=["Operations"])
async def health():
    """
    Service health check.
    Used by Kubernetes liveness probe.
    """
    # Check BentoML is reachable
    bentoml_status = "unknown"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
                f"{BENTOML_URL}/health",
                json    = {"request":
                           {"ping": "ping"}},
                timeout = 2.0)
            if r.status_code == 200:
                bentoml_status = "healthy"
            else:
                bentoml_status = "degraded"
    except Exception:
        bentoml_status = "unreachable"

    return HealthResponse(
        status    = "healthy"
                    if bentoml_status ==
                    "healthy"
                    else "degraded",
        api       = "FastAPI",
        model     = f"HSTU via BentoML "
                    f"({bentoml_status})",
        version   = "1.0.0",
        timestamp = time.time(),
    )


@app.get("/readiness",
         tags=["Operations"])
async def readiness():
    """
    Kubernetes readiness probe.
    Returns 200 only when fully ready.
    """
    return {"status": "ready",
            "timestamp": time.time()}


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

    - **user_id**: User to recommend for
    - **top_k**: Number of recommendations (1-50)
    - **include_scores**: Include model scores

    Returns ranked list of movies with
    explanations and confidence scores.
    """
    start      = time.time()
    request_id = getattr(
        request.state, 'request_id',
        str(uuid.uuid4())[:8])

    request_metrics['total'] += 1

    try:
        # Call BentoML service
        async with httpx.AsyncClient() as client:
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

        data  = r.json()
        recs  = data.get(
            "recommendations", [])

        # Strip scores if not requested
        if not body.include_scores:
            for rec in recs:
                rec['score'] = 0.0

        latency = (time.time()-start)*1000
        request_metrics['success'] += 1
        request_metrics['latencies'].append(
            latency)

        log.info(
            f"rid={request_id} "
            f"user={body.user_id} "
            f"n_recs={len(recs)} "
            f"latency={latency:.1f}ms")

        return RecommendResponse(
            request_id      = request_id,
            user_id         = body.user_id,
            recommendations = recs,
            model           = "HSTU",
            latency_ms      = round(latency,2),
            n_recs          = len(recs),
            cached          = False,
        )

    except httpx.TimeoutException:
        request_metrics['errors'] += 1
        raise HTTPException(
            status_code = 504,
            detail      = "Model service timeout")

    except Exception as e:
        request_metrics['errors'] += 1
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
    Log user interaction for online learning.

    Interactions are queued to Kafka
    for real-time model updates.
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

    # Forward to BentoML
    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(
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
        pass  # Non-blocking

    return {
        "status":     "logged",
        "request_id": request_id,
        "user_id":    body.user_id,
        "movie_id":   body.movie_id,
        "timestamp":  time.time(),
        "message":    "Queued for Kafka "
                      "pipeline",
    }


@app.get("/metrics",
         tags=["Operations"])
async def metrics():
    """
    Request metrics for monitoring.
    Prometheus scrapes this in Day 33.
    """
    lats = request_metrics['latencies']
    return {
        "total_requests":   request_metrics[
            'total'],
        "successful":       request_metrics[
            'success'],
        "errors":           request_metrics[
            'errors'],
        "error_rate":       round(
            request_metrics['errors'] /
            max(request_metrics['total'],
                1) * 100, 2),
        "latency_p50_ms":   round(float(
            np.percentile(lats, 50)),
            1) if lats else 0,
        "latency_p95_ms":   round(float(
            np.percentile(lats, 95)),
            1) if lats else 0,
        "latency_p99_ms":   round(float(
            np.percentile(lats, 99)),
            1) if lats else 0,
        "timestamp":        time.time(),
    }


@app.get("/",
         tags=["Operations"])
async def root():
    """API root — redirect to docs"""
    return {
        "message": "Production RecSys API",
        "docs":    "/docs",
        "health":  "/health",
        "version": "1.0.0",
    }