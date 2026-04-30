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


