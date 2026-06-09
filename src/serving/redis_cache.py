"""
Redis Caching Layer for Recommendations.

Features:
  → Per-user recommendation caching
  → TTL = 1 hour (configurable)
  → Cache invalidation on feedback
  → Cache warming for active users
  → Hit rate monitoring
  → JSON serialization

Usage:
  from src.serving.redis_cache import RecommendationCache
  cache = RecommendationCache()
  cache.get(user_id)
  cache.set(user_id, recs)
  cache.invalidate(user_id)
"""

import json
import time
import redis
import logging
from typing import Optional, List
from pathlib import Path

log = logging.getLogger("redis_cache")


class RecommendationCache:
    """
    Redis-backed recommendation cache.

    Key structure:
      rec:user:{user_id}     → recommendations
      rec:meta:{user_id}     → metadata
      rec:stats:hits         → hit counter
      rec:stats:misses       → miss counter

    TTL: 3600 seconds (1 hour)
    Production: Netflix uses 30min TTL
                Spotify uses 1hr TTL
    """

    def __init__(self,
                 host:    str = 'localhost',
                 port:    int = 6379,
                 db:      int = 0,
                 ttl:     int = 3600,
                 prefix:  str = 'rec'):
        self.ttl    = ttl
        self.prefix = prefix
        self.hits   = 0
        self.misses = 0

        try:
            self.redis = redis.Redis(
                host            = host,
                port            = port,
                db              = db,
                decode_responses = True,
                socket_timeout  = 2,
                socket_connect_timeout = 2,
            )
            # Test connection
            self.redis.ping()
            self.connected = True
            log.info(
                f"Redis connected: "
                f"{host}:{port}")
        except Exception as e:
            self.connected = False
            log.warning(
                f"Redis unavailable: {e}. "
                f"Running without cache.")

    def _key(self, user_id: int) -> str:
        """Build cache key for user"""
        return f"{self.prefix}:user:{user_id}"

    def _meta_key(self,
                   user_id: int) -> str:
        """Build metadata key"""
        return (f"{self.prefix}:"
                f"meta:{user_id}")

    def get(self,
            user_id: int
            ) -> Optional[List[dict]]:
        """
        Get cached recommendations.
        Returns None on cache miss.
        """
        if not self.connected:
            return None

        try:
            key  = self._key(user_id)
            data = self.redis.get(key)

            if data:
                self.hits += 1
                self.redis.incr(
                    f"{self.prefix}:"
                    f"stats:hits")
                log.debug(
                    f"Cache HIT user={user_id}")
                return json.loads(data)

            self.misses += 1
            self.redis.incr(
                f"{self.prefix}:"
                f"stats:misses")
            log.debug(
                f"Cache MISS user={user_id}")
            return None

        except Exception as e:
            log.warning(
                f"Cache get error: {e}")
            return None

    def set(self,
            user_id: int,
            recs:    List[dict],
            ttl:     Optional[int] = None
            ) -> bool:
        """
        Cache recommendations for user.
        Returns True on success.
        """
        if not self.connected:
            return False

        try:
            key      = self._key(user_id)
            meta_key = self._meta_key(user_id)
            ttl      = ttl or self.ttl

            # Store recommendations
            self.redis.setex(
                key,
                ttl,
                json.dumps(recs))

            # Store metadata
            self.redis.setex(
                meta_key,
                ttl,
                json.dumps({
                    "user_id":    user_id,
                    "n_recs":     len(recs),
                    "cached_at":  time.time(),
                    "expires_at": time.time()
                                  + ttl,
                    "ttl":        ttl,
                }))

            log.debug(
                f"Cache SET user={user_id} "
                f"n={len(recs)} ttl={ttl}s")
            return True

        except Exception as e:
            log.warning(
                f"Cache set error: {e}")
            return False

    def invalidate(self,
                    user_id: int) -> bool:
        """
        Invalidate cache for user.
        Called when user gives feedback.
        """
        if not self.connected:
            return False

        try:
            key      = self._key(user_id)
            meta_key = self._meta_key(user_id)

            deleted = self.redis.delete(
                key, meta_key)

            log.info(
                f"Cache INVALIDATED "
                f"user={user_id} "
                f"keys={deleted}")
            return deleted > 0

        except Exception as e:
            log.warning(
                f"Cache invalidate error: {e}")
            return False

    def warm(self,
              user_ids:   List[int],
              get_recs_fn) -> dict:
        """
        Pre-populate cache for active users.
        Called at service startup.

        Production: warm top 1000 users
        by interaction count.
        """
        results = {
            "warmed":  0,
            "failed":  0,
            "skipped": 0,
        }

        for uid in user_ids:
            # Skip if already cached
            if self.get(uid) is not None:
                results['skipped'] += 1
                continue

            try:
                recs = get_recs_fn(uid)
                if recs:
                    self.set(uid, recs)
                    results['warmed'] += 1
                else:
                    results['failed'] += 1
            except Exception:
                results['failed'] += 1

        log.info(
            f"Cache warm complete: "
            f"{results}")
        return results

    def stats(self) -> dict:
        """
        Get cache statistics.
        """
        if not self.connected:
            return {
                "connected":  False,
                "hits":       0,
                "misses":     0,
                "hit_rate":   0.0,
            }

        try:
            hits   = int(self.redis.get(
                f"{self.prefix}:stats:hits")
                or 0)
            misses = int(self.redis.get(
                f"{self.prefix}:stats:misses")
                or 0)
            total  = hits + misses
            hr     = hits/total*100 \
                     if total > 0 else 0.0

            # Count cached users
            pattern = (f"{self.prefix}"
                       f":user:*")
            n_cached = len(list(
                self.redis.scan_iter(
                    pattern)))

            # Redis memory info
            info = self.redis.info('memory')
            mem  = info.get(
                'used_memory_human', 'N/A')

            return {
                "connected":    True,
                "hits":         hits,
                "misses":       misses,
                "hit_rate_pct": round(hr, 1),
                "total_requests": total,
                "cached_users": n_cached,
                "memory_used":  mem,
                "ttl_seconds":  self.ttl,
            }

        except Exception as e:
            return {
                "connected": True,
                "error":     str(e)}

    def flush_all(self) -> bool:
        """
        Clear all cached recommendations.
        Use with caution in production.
        """
        if not self.connected:
            return False
        try:
            pattern = (f"{self.prefix}:*")
            keys    = list(
                self.redis.scan_iter(pattern))
            if keys:
                self.redis.delete(*keys)
            log.warning(
                f"Cache flushed: "
                f"{len(keys)} keys deleted")
            return True
        except Exception as e:
            log.warning(
                f"Cache flush error: {e}")
            return False