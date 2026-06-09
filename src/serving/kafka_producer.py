"""
Kafka Producer for User Interactions.

Publishes interaction events to Kafka
for real-time model updates.

Topics:
  user-interactions  → all user events
  recommendations    → rec requests logged
  dead-letter        → failed events

Usage:
  from src.serving.kafka_producer import
      InteractionProducer
  producer = InteractionProducer()
  producer.send_interaction(
      user_id=1, movie_id=356,
      rating=4.5, action="watch")
"""

import json
import time
import logging
import uuid
from typing import Optional
from datetime import datetime

log = logging.getLogger("kafka_producer")

# Topics
TOPIC_INTERACTIONS  = "user-interactions"
TOPIC_RECS          = "recommendations"
TOPIC_DEAD_LETTER   = "dead-letter"


class InteractionProducer:
    """
    Kafka producer for user interactions.

    Sends structured events to Kafka
    for downstream consumption by:
    → Model update consumer (Day 32)
    → Analytics pipeline
    → A/B test logging
    """

    def __init__(self,
                 bootstrap_servers: str =
                     'localhost:9092'):
        self.bootstrap_servers = \
            bootstrap_servers
        self.connected = False
        self.producer  = None
        self._connect()

    def _connect(self):
        """Connect to Kafka broker"""
        try:
            from kafka import KafkaProducer
            self.producer = KafkaProducer(
                bootstrap_servers =
                    self.bootstrap_servers,
                value_serializer  = lambda v:
                    json.dumps(v).encode(
                        'utf-8'),
                key_serializer    = lambda k:
                    str(k).encode('utf-8'),
                acks              = 'all',
                retries           = 3,
                max_block_ms      = 5000,
            )
            self.connected = True
            log.info(
                f"Kafka connected: "
                f"{self.bootstrap_servers}")
        except Exception as e:
            self.connected = False
            log.warning(
                f"Kafka unavailable: {e}. "
                f"Events will be logged only.")

    def _build_event(self,
                      event_type: str,
                      **kwargs) -> dict:
        """Build structured event"""
        return {
            "event_id":   str(uuid.uuid4()),
            "event_type": event_type,
            "timestamp":  time.time(),
            "datetime":   datetime.now(
            ).isoformat(),
            **kwargs,
        }

    def send_interaction(
            self,
            user_id:   int,
            movie_id:  int,
            rating:    float,
            action:    str   = "watch",
            watch_pct: Optional[float] = None,
            ) -> bool:
        """
        Send user interaction event.
        Called by FastAPI /feedback endpoint.
        """
        event = self._build_event(
            event_type = "interaction",
            user_id    = user_id,
            movie_id   = movie_id,
            rating     = rating,
            action     = action,
            watch_pct  = watch_pct,
        )

        return self._send(
            TOPIC_INTERACTIONS,
            key   = user_id,
            value = event)

    def send_recommendation_logged(
            self,
            user_id:   int,
            recs:      list,
            model:     str,
            latency_ms: float,
            cached:    bool) -> bool:
        """
        Log recommendation event.
        Used for A/B testing + analytics.
        """
        event = self._build_event(
            event_type  = "recommendation",
            user_id     = user_id,
            n_recs      = len(recs),
            model       = model,
            latency_ms  = latency_ms,
            cached      = cached,
            movie_ids   = [
                r.get('movie_id')
                for r in recs[:5]],
        )

        return self._send(
            TOPIC_RECS,
            key   = user_id,
            value = event)

    def _send(self,
              topic: str,
              key:   int,
              value: dict) -> bool:
        """
        Send event to Kafka topic.
        Falls back to logging if unavailable.
        """
        if not self.connected or \
                self.producer is None:
            log.info(
                f"[KAFKA LOG] topic={topic} "
                f"key={key} "
                f"event={value['event_type']}")
            return False

        try:
            future = self.producer.send(
                topic,
                key   = key,
                value = value)

            # Wait for ack with timeout
            record = future.get(timeout=5)

            log.info(
                f"Kafka sent: "
                f"topic={topic} "
                f"partition={record.partition} "
                f"offset={record.offset} "
                f"key={key}")
            return True

        except Exception as e:
            log.warning(
                f"Kafka send failed: {e}")
            # Send to dead letter queue
            self._dead_letter(
                topic, key, value, str(e))
            return False

    def _dead_letter(self,
                      original_topic: str,
                      key: int,
                      value: dict,
                      error: str):
        """Send failed event to DLQ"""
        if not self.producer:
            return
        try:
            dlq_event = {
                "original_topic": original_topic,
                "original_event": value,
                "error":          error,
                "timestamp":      time.time(),
            }
            self.producer.send(
                TOPIC_DEAD_LETTER,
                key   = key,
                value = dlq_event)
        except Exception:
            pass  # DLQ also failed — just log

    def flush(self):
        """Flush all pending messages"""
        if self.producer:
            self.producer.flush()

    def close(self):
        """Clean shutdown"""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            log.info("Kafka producer closed")