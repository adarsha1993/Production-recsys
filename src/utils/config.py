# src/utils/config.py
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

def load_domain_config(domain: str) -> dict:
    config_path = Path(f"configs/domains/{domain}.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)

class Settings:
    # AWS
    AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION            = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
    S3_BUCKET             = os.getenv("S3_BUCKET_NAME")

    # PostgreSQL
    POSTGRES_HOST     = os.getenv("POSTGRES_HOST", "localhost")
    POSTGRES_PORT     = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_DB       = os.getenv("POSTGRES_DB", "recsys_db")
    POSTGRES_USER     = os.getenv("POSTGRES_USER", "recsys_user")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "recsys_user")

    # Redis
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

    # Qdrant
    QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))

    # MLflow
    MLFLOW_TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "recsys_experiments")

    # TMDB
    TMDB_API_KEY  = os.getenv("TMDB_API_KEY")
    TMDB_BASE_URL = os.getenv("TMDB_BASE_URL", "https://api.themoviedb.org/3")

    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    # Kafka
    KAFKA_SERVERS     = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_TOPIC       = os.getenv("KAFKA_TOPIC_USER_EVENTS", "user_interactions")

    # App
    ENV       = os.getenv("APP_ENV", "development")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

settings = Settings()

def get_ratings_path(use_full: bool = False) -> str:
    """
    Always use ratings_small for dev.
    Only switch to ratings.csv explicitly with use_full=True.
    ratings.csv is 26M rows — NEVER load with pd.read_csv directly.
    Always use PySpark for full dataset.
    """
    if use_full:
        print("⚠️  Loading full ratings.csv (26M rows) — use PySpark only")
        return "data/raw/ratings.csv"
    return "data/raw/ratings_small.csv"