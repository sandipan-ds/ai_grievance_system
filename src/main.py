"""
Database connection test for AI Grievance System.

This script tests connectivity to the Supabase PostgreSQL database
by loading environment credentials and attempting a connection.
"""

import logging
import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Environment variable names
ENV_VARS = {
    "user": "user",
    "password": "password",
    "host": "host",
    "port": "port",
    "dbname": "dbname",
}


def load_env_variables() -> dict[str, str]:
    """
    Load and validate environment variables from .env file.
    
    Returns:
        dict: Environment variables as key-value pairs
        
    Raises:
        ValueError: If any required environment variable is missing
    """
    load_dotenv()
    
    env_vars = {}
    missing_vars = []
    
    for key, env_name in ENV_VARS.items():
        value = os.getenv(env_name)
        if not value:
            missing_vars.append(env_name)
        else:
            env_vars[key] = value
    
    if missing_vars:
        raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
    
    return env_vars


def create_connection(env_vars: dict[str, str]) -> Engine:
    """
    Create SQLAlchemy database engine with loaded credentials.
    
    Args:
        env_vars: Dictionary containing database connection parameters
        
    Returns:
        Engine: SQLAlchemy engine instance
    """
    database_url = (
        f"postgresql+psycopg2://{env_vars['user']}:{env_vars['password']}"
        f"@{env_vars['host']}:{env_vars['port']}/{env_vars['dbname']}?sslmode=require"
    )
    
    engine = create_engine(database_url)
    # If using Transaction Pooler or Session Pooler, uncomment to disable client-side pooling:
    # from sqlalchemy.pool import NullPool
    # engine = create_engine(database_url, poolclass=NullPool)
    
    return engine


def test_connection(engine: Engine) -> bool:
    """
    Test the database connection.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        with engine.connect() as connection:
            logger.info("✅ Connection successful!")
            return True
    except Exception as e:
        logger.error(f"❌ Failed to connect: {type(e).__name__}: {e}")
        return False


def main() -> int:
    """
    Main entry point for connection test.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        logger.info("Loading environment variables...")
        env_vars = load_env_variables()
        
        logger.info("Creating database engine...")
        engine = create_connection(env_vars)
        
        logger.info("Testing connection...")
        success = test_connection(engine)
        
        engine.dispose()
        
        return 0 if success else 1
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

from src.db.postgres import create_postgres_engine, test_connection
from src.logger.logger import append_prediction_log
from src.ml.model_loader import ModelBundle, load_model_bundle
from src.ml.predictor import predict_complaint
from src.schemas.schemas import PredictionRequest, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.models = load_model_bundle()

    try:
        app.state.db_engine = create_postgres_engine()
        app.state.db_connected = test_connection(app.state.db_engine)
    except Exception:
        app.state.db_engine = None
        app.state.db_connected = False

    yield


app = FastAPI(
    title="AI Grievance System API",
    version="v1.0",
    docs_url="/docs",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    payload: PredictionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PredictionResponse:
    try:
        models: ModelBundle = request.app.state.models
        predicted_department, severity = predict_complaint(models, payload.complaint)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc

    background_tasks.add_task(
        append_prediction_log,
        complaint=payload.complaint,
        predicted_department=predicted_department,
        severity=severity,
        model_version="v1.0",
    )

    return PredictionResponse(
        predicted_department=predicted_department,
        severity=severity,
    )
