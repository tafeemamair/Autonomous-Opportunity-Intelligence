from datetime import datetime
from enum import StrEnum

from pydantic import Field, HttpUrl

from .common import AOIBaseModel, Evidence
from .opportunity import DecisionMaker


class ResearchStatus(StrEnum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class ResearchResult(AOIBaseModel):
    """Structured research output for a single candidate entity."""

    candidate_id: str
    company_name: str
    website: HttpUrl | None = None
    location: str | None = None
    industry: str | None = None
    people: list[DecisionMaker] = Field(default_factory=list)
    technology_signals: list[str] = Field(default_factory=list)
    business_signals: list[str] = Field(default_factory=list)
    problem_signals: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    research_status: ResearchStatus = ResearchStatus.COMPLETED
    researched_at: datetime
