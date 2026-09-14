from datetime import UTC, datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..config import get_settings

settings = get_settings()
engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db(bind_engine=None) -> None:
    """Initialize database tables using existing Base metadata."""
    from . import models  # noqa: F401
    from .base import Base

    Base.metadata.create_all(bind=bind_engine or engine)


def persist_run(
    run_id: str,
    status: str,
    objective: dict,
    statistics: dict | None = None,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
    session_factory=None,
) -> None:
    """Persist run-level execution information into the existing runs table."""
    from .models import Run

    factory = session_factory or SessionLocal
    with factory() as session:
        try:
            run_record = session.get(Run, run_id)
        except Exception:
            session.rollback()
            init_db(bind_engine=session.get_bind())
            run_record = session.get(Run, run_id)
        if run_record is None:
            run_record = Run(
                id=run_id,
                status=status,
                objective=objective,
                statistics=statistics or {},
                started_at=started_at or datetime.now(UTC),
                completed_at=completed_at,
            )
            session.add(run_record)
        else:
            run_record.status = status
            run_record.objective = objective
            run_record.statistics = statistics or {}
            if completed_at is not None:
                run_record.completed_at = completed_at
        session.commit()
