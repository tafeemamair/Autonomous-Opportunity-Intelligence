from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from aoi.db.base import Base
from aoi.db.models import Company


def test_database_models_create_tables():
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        company = Company(name="Example AI", domain="example.com")
        session.add(company)
        session.commit()
        assert company.id is not None
