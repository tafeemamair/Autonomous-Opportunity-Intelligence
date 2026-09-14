from datetime import datetime

from .common import AOIBaseModel, Priority, RunStatus, SourceType, VerificationStatus
from .qualification import ProspectType, QualificationStatus
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
