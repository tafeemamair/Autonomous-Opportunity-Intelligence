from datetime import datetime
from enum import StrEnum

from pydantic import Field, HttpUrl

from .common import AOIBaseModel, Evidence
from .opportunity import DecisionMaker


class ResearchStatus(StrEnum):
    COMPLETED = "COMPLETED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


class ResearchQuality(AOIBaseModel):
    """Quality and completeness evaluation of gathered research."""

    completeness_score: float = Field(default=0.0, ge=0.0, le=100.0)
    evidence_diversity_score: float = Field(default=0.0, ge=0.0, le=100.0)
    recency_score: float = Field(default=0.0, ge=0.0, le=100.0)
    signal_depth_score: float = Field(default=0.0, ge=0.0, le=100.0)
    overall_quality_score: float = Field(default=0.0, ge=0.0, le=100.0)


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
    research_quality: ResearchQuality | None = None
    queries_executed: int = 0
    queries_saved: int = 0
    is_saturated: bool = False
    researched_at: datetime


class ResearchEvaluation(AOIBaseModel):
    """Measurable evaluation metrics across an entire research execution batch."""

    candidates_researched_count: int = 0
    total_queries_executed: int = 0
    total_queries_saved: int = 0
    cost_savings_percentage: float = 0.0
    evidence_yield_per_query: float = 0.0
    saturation_rate: float = 0.0
    tier1_source_ratio: float = 0.0
    decision_maker_discovery_rate: float = 0.0
    average_quality_score: float = 0.0
    evaluated_at: datetime
