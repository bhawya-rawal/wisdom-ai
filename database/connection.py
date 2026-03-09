import logging
from typing import Generator
from sqlmodel import SQLModel, create_engine, Session
from config.settings import settings

logger = logging.getLogger(__name__)

# Choose engine based on connection URL
db_url = settings.DATABASE_URL
connect_args = {}

# Check if SQLite fallback is requested/needed
if db_url.startswith("sqlite") or settings.USE_SQLITE_FALLBACK:
    db_url = settings.SQLITE_URL
    connect_args = {"check_same_thread": False}
    logger.warning("Using SQLite database fallback: %s", db_url)
else:
    # Postgres specific configurations
    connect_args = {
        # Neon and modern postgres engines benefit from sslmode=require in production,
        # but for local dev docker containers we might omit it if needed.
        # Handled in the connection string.
    }
    logger.info("Connecting to PostgreSQL database...")

try:
    engine = create_engine(db_url, echo=False, connect_args=connect_args)
except Exception as e:
    logger.error("Failed to initialize primary database engine: %s. Falling back to SQLite.", e)
    engine = create_engine(settings.SQLITE_URL, echo=False, connect_args={"check_same_thread": False})

def init_db() -> None:
    """Initialize database tables"""
    logger.info("Initializing database schemas...")
    # SQLModel will bind to the metadata and create tables if they do not exist
    SQLModel.metadata.create_all(engine)

def get_session() -> Generator[Session, None, None]:
    """Dependency injector for FastAPI endpoints to get active db sessions"""
    with Session(engine) as session:
        yield session
