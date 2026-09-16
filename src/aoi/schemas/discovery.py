from datetime import datetime
from enum import StrEnum

from pydantic import Field, HttpUrl

from .common import AOIBaseModel, SourceType


class CandidateResearchPriority(StrEnum):
    URGENT = "URGENT"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class DiscoveryBudget(AOIBaseModel):
    """Explicit bounded budget controlling discovery execution limits."""

    max_queries: int = Field(default=25, ge=1, le=500)
    max_results_per_query: int = Field(default=5, ge=1, le=100)
    max_candidates: int = Field(default=50, ge=1, le=1000)


class DiscoveryStrategy(AOIBaseModel):
    name: str
    purpose: str
    query_templates: list[str] = Field(default_factory=list)
    signal_category: str = "general"
    market: str | None = None
    opportunity: str | None = None
    target_company_type: str | None = None
    expected_signals: list[str] = Field(default_factory=list)


class DiscoveryPlan(AOIBaseModel):
    strategies: list[DiscoveryStrategy]
    created_at: datetime
    budget: DiscoveryBudget = Field(default_factory=DiscoveryBudget)


class Candidate(AOIBaseModel):
    candidate_id: str
    company_name: str
    website: HttpUrl | None = None
    location: str | None = None
    discovery_strategy: str
    discovery_reason: str
    source_urls: list[HttpUrl] = Field(default_factory=list)
    source_types: list[SourceType] = Field(default_factory=list)
    discovered_at: datetime
    normalized_name: str | None = None
    domain: str | None = None
    research_priority_score: float = Field(default=50.0, ge=0.0, le=100.0)
    research_priority: CandidateResearchPriority = CandidateResearchPriority.MEDIUM
    signal_categories: list[str] = Field(default_factory=list)
    discovery_reasons: list[str] = Field(default_factory=list)


class DiscoveryEvaluation(AOIBaseModel):
    candidates_discovered_count: int = 0
    unique_companies_count: int = 0
    deduplication_rate: float = 0.0
    multi_signal_candidate_count: int = 0
    multi_signal_ratio: float = 0.0
    strategy_yield: dict[str, int] = Field(default_factory=dict)
    channel_diversity_score: float = 0.0
    evaluated_at: datetime
    budget: DiscoveryBudget | None = None
    queries_executed: int = 0
    queries_capped: bool = False
