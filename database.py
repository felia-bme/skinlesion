"""
Database connection and schema for prediction history storage.
Uses SQLite for a lightweight PoC; swap DATABASE_URL for production.
"""

import os
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class PredictionHistory(Base):
    """Stores each prediction request for audit / review."""

    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    filename = Column(String(255), nullable=False)
    top_dx = Column(String(32), nullable=False)
    top_label = Column(String(128), nullable=False)
    top_probability = Column(Float, nullable=False)
    characteristic = Column(String(32), nullable=False)
    predictions_json = Column(Text, nullable=False)  # full top-k payload as JSON


def init_db() -> None:
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """Yield a DB session (FastAPI dependency style)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_prediction(
    filename: str,
    top_dx: str,
    top_label: str,
    top_probability: float,
    characteristic: str,
    predictions_json: str,
) -> PredictionHistory:
    """Insert one prediction record and return it."""
    db = SessionLocal()
    try:
        row = PredictionHistory(
            filename=filename,
            top_dx=top_dx,
            top_label=top_label,
            top_probability=top_probability,
            characteristic=characteristic,
            predictions_json=predictions_json,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row
    finally:
        db.close()


def list_recent(limit: int = 20) -> list:
    """Return the most recent prediction rows."""
    db = SessionLocal()
    try:
        return (
            db.query(PredictionHistory)
            .order_by(PredictionHistory.timestamp.desc())
            .limit(limit)
            .all()
        )
    finally:
        db.close()
