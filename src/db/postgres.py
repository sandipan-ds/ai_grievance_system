from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


SRC_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = SRC_DIR / ".env"


def load_db_settings(env_path: Path | None = None) -> dict[str, str]:
    """Load database settings from the src-local .env file."""
    load_dotenv(env_path or DEFAULT_ENV_PATH, override=False)

    return {
        "user": os.getenv("user", ""),
        "password": os.getenv("password", ""),
        "host": os.getenv("host", ""),
        "port": os.getenv("port", ""),
        "dbname": os.getenv("dbname", ""),
    }


def build_database_url(settings: dict[str, str] | None = None) -> str:
    settings = settings or load_db_settings()
    return (
        "postgresql+psycopg2://"
        f"{settings['user']}:{settings['password']}@"
        f"{settings['host']}:{settings['port']}/{settings['dbname']}?sslmode=require"
    )


def create_postgres_engine(database_url: str | None = None) -> Engine:
    return create_engine(database_url or build_database_url())


def test_connection(engine: Engine | None = None) -> bool:
    db_engine = engine or create_postgres_engine()
    with db_engine.connect():
        return True
