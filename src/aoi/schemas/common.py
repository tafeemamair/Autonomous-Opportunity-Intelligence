from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class AOIBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RunStatus(StrEnum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    DISCOVERING = "DISCOVERING"
    RESEARCHING = "RESEARCHING"
    QUALIFYING = "QUALIFYING"
    VERIFYING = "VERIFYING"
    SCORING = "SCORING"
    REPORTING = "REPORTING"
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class Priority(StrEnum):
    HIGH = "HIGH"
    QUALIFIED = "QUALIFIED"
    WATCHLIST = "WATCHLIST"
    DISCARD = "DISCARD"


class VerificationStatus(StrEnum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"


class SourceType(StrEnum):
    COMPANY_WEBSITE = "company_website"
    OFFICIAL_JOB = "official_job"
    OFFICIAL_ANNOUNCEMENT = "official_announcement"
    REGULATORY = "regulatory"
    NEWS = "news"
    INDUSTRY_PUBLICATION = "industry_publication"
    LINKEDIN = "linkedin"
    JOB_BOARD = "job_board"
    DIRECTORY = "directory"
    SOCIAL = "social"
    OTHER = "other"


class Source(AOIBaseModel):
    url: HttpUrl
    source_type: SourceType
    title: str | None = None
    publisher: str | None = None
    published_at: datetime | None = None
    accessed_at: datetime
    reliability: int = Field(ge=0, le=100)


class Evidence(AOIBaseModel):
    claim: str = Field(min_length=1)
    source: Source
    evidence_summary: str = Field(min_length=1)
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED
    recency_score: int = Field(default=50, ge=0, le=100)
