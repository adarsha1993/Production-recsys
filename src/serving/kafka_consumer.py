"""
Kafka Consumer for Model Online Updates.

Reads user interaction events from Kafka
and applies lightweight model updates.

Usage:
  python src/serving/kafka_consumer.py

Or as background service:
  from src.serving.kafka_consumer import
      InteractionConsumer
  consumer = InteractionConsumer()
  consumer.start()
"""

import sys
import json
import time
import logging
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from typing import Optional
from collections import defaultdict

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))

logging.basicConfig(
    level  = logging.INFO,
    format = '%(asctime)s %(levelname)s '
              '%(name)s %(message)s')
log = logging.getLogger("kafka_consumer")

TOPIC_INTERACTIONS = "user-interactions"
TOPIC_DEAD_LETTER  = "dead-letter"
CKPT = BASE / 'models' / 'checkpoints'


class InteractionConsumer:
    """
    Kafka consumer that processes
    user interaction events and applies
    lightweight model updates.

    Implements TikTok Monolith pattern:
    User clicks → Kafka → embedding update
    All within <100ms end-to-end.
    """

    def __init__(self,
                 bootstrap_servers: str =
                     'localhost:9092',
                 group_id: str =
                     'recsys-consumer',
                 update_every: int = 5):
        self.bootstrap_servers = \
            bootstrap_servers
        self.group_id     = group_id
        self.update_every = update_every
        self.event_buffer = []
        self.update_count = 0
        self.error_count  = 0
        self.connected    = False
        self.consumer     = None

        # Load model for updates
        self._load_model()
        self._connect()

    def _load_model(self):
        """Load HSTU model for updates"""
        try:
            self.vocab = joblib.load(
                CKPT / 'item_vocabulary.joblib')
            self.user_seqs = joblib.load(
                CKPT / 'user_sequences.joblib')

            PAD_TOKEN  = self.vocab['PAD']
            vocab_size = self.vocab['vocab_size']

            # Import HSTU
            from src.serving.bentoml_service \
                import HSTURanker

            self.model = HSTURanker(
                vocab_size = vocab_size,
                pad_token  = PAD_TOKEN,
                offset     = 3,
            )
            self.model.load_state_dict(
                torch.load(
                    CKPT / 'hstu_best.pt',
                    map_location='cpu',
                    weights_only=True))
            self.model.train()

            # Small LR for online updates
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=1e-4)

            self.movie2token = \
                self.vocab['movie2token']
            self.PAD_TOKEN   = PAD_TOKEN

            log.info("✅ Model loaded for "
                     "online updates")
        except Exception as e:
            log.warning(
                f"Model load failed: {e}")
            self.model = None

    def _connect(self):
        """Connect to Kafka"""
        try:
            from kafka import KafkaConsumer
            self.consumer = KafkaConsumer(
                TOPIC_INTERACTIONS,
                bootstrap_servers =
                    self.bootstrap_servers,
                group_id          =
                    self.group_id,
                value_deserializer = lambda v:
                    json.loads(
                        v.decode('utf-8')),
                auto_offset_reset  = 'latest',
                enable_auto_commit = True,
                consumer_timeout_ms = 1000,
            )
            self.connected = True
            log.info(
                f"Kafka consumer connected: "
                f"{self.bootstrap_servers}")
        except Exception as e:
            self.connected = False
            log.warning(
                f"Kafka unavailable: {e}")

    def process_event(self,
                       event: dict) -> bool:
        """
        Process single interaction event.
        Apply lightweight gradient update.
        """
        if self.model is None:
            return False

        try:
            user_id  = event.get('user_id')
            movie_id = event.get('movie_id')
            rating   = float(event.get(
                'rating', 3.5))

            seq = self.user_seqs.get(
                user_id, [])
            if not seq:
                return False

            tok = self.movie2token.get(movie_id)
            if not tok:
                return False

            # Build input
            pad_l = 50 - len(seq)
            hist  = torch.LongTensor(
                [[self.PAD_TOKEN]*pad_l +
                  seq[-50:]])

            # Positive = interacted movie
            pos = torch.LongTensor([tok])

            # Negative = random
            import numpy as np
            all_toks = list(
                self.movie2token.values())
            neg_tok  = np.random.choice(
                all_toks)
            neg = torch.LongTensor([neg_tok])

            # Rating label
            r = torch.FloatTensor([rating])
            c = torch.FloatTensor(
                [min(1.0,
                     (rating-0.5)/4.5)])

            # Forward + loss
            from src.serving.bentoml_service \
                import HSTURanker

            self.model.train()
            u      = self.model.encode_user(
                hist)
            p_emb  = self.model.item_emb(pos)
            n_emb  = self.model.item_emb(neg)
            ps     = (u * p_emb).sum(-1)
            ns     = (u * n_emb).sum(-1)
            loss   = -F.logsigmoid(
                ps - ns).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update sequence
            self.user_seqs[user_id] = \
                seq[-49:] + [tok]

            self.update_count += 1
            log.info(
                f"Updated user={user_id} "
                f"movie={movie_id} "
                f"loss={loss.item():.4f} "
                f"total_updates="
                f"{self.update_count}")
            return True

        except Exception as e:
            self.error_count += 1
            log.warning(
                f"Process event failed: {e}")
            return False

    def start(self,
               max_events: Optional[int] = None):
        """
        Start consuming events.
        Runs until interrupted or max_events.
        """
        if not self.connected:
            log.warning(
                "Kafka not connected. "
                "Simulating consumer.")
            return self._simulate()

        log.info(
            f"Consumer started. "
            f"Topic: {TOPIC_INTERACTIONS}")

        processed = 0
        try:
            for message in self.consumer:
                event = message.value
                start = time.time()

                success = self.process_event(
                    event)

                elapsed = (
                    time.time()-start)*1000

                log.info(
                    f"Event processed: "
                    f"success={success} "
                    f"latency={elapsed:.1f}ms")

                processed += 1
                if max_events and \
                        processed >= max_events:
                    break

        except KeyboardInterrupt:
            log.info("Consumer stopped")
        finally:
            if self.consumer:
                self.consumer.close()

        return {
            "processed":    processed,
            "updates":      self.update_count,
            "errors":       self.error_count,
        }

    def _simulate(self) -> dict:
        """
        Simulate consumer without Kafka.
        Uses test events for demonstration.
        """
        log.info("Running consumer simulation")

        test_events = [
            {
                "event_type": "interaction",
                "user_id":    1,
                "movie_id":   356,
                "rating":     4.5,
                "action":     "watch",
                "timestamp":  time.time(),
            },
            {
                "event_type": "interaction",
                "user_id":    2,
                "movie_id":   296,
                "rating":     3.0,
                "action":     "watch",
                "timestamp":  time.time(),
            },
            {
                "event_type": "interaction",
                "user_id":    3,
                "movie_id":   318,
                "rating":     5.0,
                "action":     "like",
                "timestamp":  time.time(),
            },
            {
                "event_type": "interaction",
                "user_id":    4,
                "movie_id":   260,
                "rating":     4.0,
                "action":     "watch",
                "timestamp":  time.time(),
            },
            {
                "event_type": "interaction",
                "user_id":    5,
                "movie_id":   593,
                "rating":     4.5,
                "action":     "share",
                "timestamp":  time.time(),
            },
        ]

        results = []
        for event in test_events:
            start   = time.time()
            success = self.process_event(event)
            elapsed = (time.time()-start)*1000

            results.append({
                "user_id":   event['user_id'],
                "movie_id":  event['movie_id'],
                "success":   success,
                "latency_ms": round(elapsed, 1),
            })

            log.info(
                f"Simulated event: "
                f"user={event['user_id']} "
                f"movie={event['movie_id']} "
                f"success={success} "
                f"latency={elapsed:.1f}ms")

        return {
            "mode":      "simulation",
            "processed": len(test_events),
            "updates":   self.update_count,
            "errors":    self.error_count,
            "results":   results,
        }


if __name__ == "__main__":
    consumer = InteractionConsumer()
    result   = consumer.start()
    print(json.dumps(result, indent=2))