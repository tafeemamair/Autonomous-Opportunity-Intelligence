from datetime import datetime

from pydantic import Field

from .common import AOIBaseModel, Priority, RunStatus, SourceType, VerificationStatus
from .discovery import DiscoveryEvaluation
from .intelligence import IntelligenceQuality, IntelligenceQualityStatus, OpportunityNarrative
from .qualification import ProspectType, QualificationStatus
from .research import ResearchEvaluation
from .scoring import ScoreBreakdown


class ReportEvidence(AOIBaseModel):
    evidence_id: str
    claim: str
    source: str
    source_type: SourceType
    published_at: datetime | None
    accessed_at: datetime | None
    evidence_summary: str
    reliability: float
    recency: float
    verification_status: str


class ReportOpportunity(AOIBaseModel):
    rank: int
    candidate_id: str
    company_name: str
    website: str | None

    opportunity_summary: str
    problem: str | None
    potential_solution: str | None

    prospect_type: ProspectType
    qualification_status: QualificationStatus
    verification_status: VerificationStatus

    opportunity_score: float
    confidence_score: float
    priority: Priority

    score_breakdown: ScoreBreakdown

    evidence: list[ReportEvidence]

    scoring_reasons: list[str]
    warnings: list[str]

    recommended_action: str

    quality: IntelligenceQuality | None = None
    narrative: OpportunityNarrative | None = None
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    converging_signals: list[str] = Field(default_factory=list)
    quality_status: IntelligenceQualityStatus | None = None


class ReportStatistics(AOIBaseModel):
    candidates_discovered: int
    candidates_researched: int
    candidates_qualified: int
    candidates_verified: int
    opportunities_scored: int

    high_priority: int
    qualified_priority: int
    watchlist_priority: int
    discard_priority: int

    discovery_evaluation: DiscoveryEvaluation | None = None
    research_evaluation: ResearchEvaluation | None = None
    queries_executed: int = 0
    queries_saved: int = 0
    cost_savings_percentage: float = 0.0
    multi_signal_candidates: int = 0


class ReportSummary(AOIBaseModel):
    headline: str
    overview: str
    top_opportunity_count: int
    high_confidence_count: int
    warning_count: int


class AOIReport(AOIBaseModel):
    run_id: str
    objective: dict
    run_status: RunStatus
    generated_at: datetime

    summary: ReportSummary
    statistics: ReportStatistics

    opportunities: list[ReportOpportunity]
    watchlist: list[ReportOpportunity]

    discarded_count: int

    warnings: list[str]
