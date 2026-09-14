from datetime import datetime

from pydantic import Field

from .common import AOIBaseModel, Priority, VerificationStatus


class ScoreBreakdown(AOIBaseModel):
    """Detailed numeric component scores and final Opportunity & Confidence scores."""

    need_fit: float = Field(ge=0.0, le=100.0)
    capability_fit: float = Field(ge=0.0, le=100.0)
    evidence_strength: float = Field(ge=0.0, le=100.0)
    timing: float = Field(ge=0.0, le=100.0)
    commercial_potential: float = Field(ge=0.0, le=100.0)
    accessibility: float = Field(ge=0.0, le=100.0)
    opportunity_score: float = Field(ge=0.0, le=100.0)
    confidence_score: float = Field(ge=0.0, le=100.0)
    priority: Priority


class ScoringResult(AOIBaseModel):
    """Structured decision output of the Opportunity Scoring Agent."""

    candidate_id: str
    company_name: str
    scores: ScoreBreakdown
    scoring_reasons: list[str] = Field(default_factory=list)
    scoring_warnings: list[str] = Field(default_factory=list)
    verification_status: VerificationStatus
    scored_at: datetime
