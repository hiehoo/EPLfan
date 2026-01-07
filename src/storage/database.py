"""Database connection and operations."""
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.config.settings import settings
from src.storage.models import Base

logger = logging.getLogger(__name__)


class Database:
    """Database connection manager."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or settings.database_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.engine = create_engine(f"sqlite:///{self.db_path}")
        self.SessionLocal = sessionmaker(bind=self.engine)

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(self.engine)
        logger.info(f"Database tables created at {self.db_path}")

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around operations."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


# Singleton instance
db = Database()
