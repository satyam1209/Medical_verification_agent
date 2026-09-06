"""Database engine, initialization, and session helpers for medverify.db."""

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from schema import Base

DATABASE_URL = "sqlite:///medverify.db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_engine():
    """Create a SQLite engine pointing to 'medverify.db'."""
    return engine


def init_db():
    """Create the medical_records table if it doesn't exist."""
    Base.metadata.create_all(engine)


def get_session():
    """Return a SQLAlchemy session."""
    return SessionLocal()


def now_str() -> str:
    """Return the current date/time as a string for ingestion_date."""
    return datetime.now().isoformat(timespec="seconds")