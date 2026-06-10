"""
Locust load test for Production RecSys API.

Usage:
  locust -f tests/load/locustfile.py
         --host http://localhost:8000
         --users 50
         --spawn-rate 5
         --run-time 60s
         --headless

Or with UI:
  locust -f tests/load/locustfile.py
         --host http://localhost:8000
"""

import random
from locust import (
    HttpUser,
    task,
    between,
    events)


class RecSysUser(HttpUser):
    """
    Simulates a real user of the
    recommendation API.

    Wait 1-3 seconds between requests
    — realistic think time.
    """
    wait_time = between(1, 3)

    def on_start(self):
        """Pick a random user ID"""
        self.user_id = random.randint(
            1, 670)

    # ── Recommend — 60% of traffic ────────
    @task(6)
    def get_recommendations(self):
        with self.client.post(
                "/recommend",
                json={
                    "user_id": self.user_id,
                    "top_k":   10,
                },
                catch_response=True) as r:
            if r.status_code == 200:
                data = r.json()
                if data.get('n_recs', 0) > 0:
                    r.success()
                else:
                    r.failure(
                        "No recs returned")
            else:
                r.failure(
                    f"Status {r.status_code}")

    # ── Feedback — 20% of traffic ─────────
    @task(2)
    def send_feedback(self):
        movie_ids = [
            356, 296, 318, 593,
            260, 480, 110, 589]
        with self.client.post(
                "/feedback",
                json={
                    "user_id":  self.user_id,
                    "movie_id": random.choice(
                        movie_ids),
                    "rating":   random.choice(
                        [3.0, 3.5, 4.0,
                         4.5, 5.0]),
                    "action":   random.choice(
                        ["watch", "like",
                         "rate"]),
                },
                catch_response=True) as r:
            if r.status_code == 200:
                r.success()
            else:
                r.failure(
                    f"Status {r.status_code}")

    # ── Health — 10% of traffic ───────────
    @task(1)
    def health_check(self):
        with self.client.get(
                "/health",
                catch_response=True) as r:
            if r.status_code == 200:
                data = r.json()
                if data.get(
                        'status') == 'healthy':
                    r.success()
                else:
                    r.failure("Degraded")
            else:
                r.failure(
                    f"Status {r.status_code}")

    # ── Metrics — 10% of traffic ──────────
    @task(1)
    def get_metrics(self):
        self.client.get("/metrics")


@events.request.add_listener
def on_request(
        request_type,
        name,
        response_time,
        response_length,
        response,
        context,
        exception,
        **kwargs):
    """Log slow requests"""
    if response_time > 500:
        print(
            f"SLOW: {name} "
            f"{response_time:.0f}ms")