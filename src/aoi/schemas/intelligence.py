from datetime import datetime
from enum import StrEnum

from pydantic import Field

from .common import AOIBaseModel


class IntelligenceQualityStatus(StrEnum):
    """Categorical quality tier based on overall Quality Score.

    HIGH: 80–100
    GOOD: 65–79
    LIMITED: 50–64
    INSUFFICIENT: <50
    """

    HIGH = "HIGH"
    GOOD = "GOOD"
    LIMITED = "LIMITED"
    INSUFFICIENT = "INSUFFICIENT"

    @classmethod
    def from_score(cls, score: float) -> "IntelligenceQualityStatus":
        """Deterministic mapping of numeric quality score to quality status."""
        if score >= 80.0:
            return cls.HIGH
        elif score >= 65.0:
            return cls.GOOD
        elif score >= 50.0:
            return cls.LIMITED
        else:
            return cls.INSUFFICIENT


class IntelligenceQuality(AOIBaseModel):
    """Detailed numeric component quality scores and overall Quality Score."""

    completeness_score: float = Field(ge=0.0, le=100.0)
    evidence_coverage_score: float = Field(ge=0.0, le=100.0)
    consistency_score: float = Field(ge=0.0, le=100.0)
    actionability_score: float = Field(ge=0.0, le=100.0)
    quality_score: float = Field(ge=0.0, le=100.0)
    warnings: list[str] = Field(default_factory=list)


class OpportunityNarrative(AOIBaseModel):
    """Deterministic, structured narrative explaining the opportunity and evidence."""

    why_company: str
    why_now: str
    identified_problem: str | None = None
    why_fit: str
    evidence_summary: str
    next_human_step: str


class IntelligenceQualityResult(AOIBaseModel):
    """Structured decision and depth evaluation of an opportunity's intelligence."""

    candidate_id: str
    company_name: str
    quality: IntelligenceQuality
    narrative: OpportunityNarrative
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    converging_signals: list[str] = Field(default_factory=list)
    material_claims_covered: int = Field(ge=0)
    material_claims_total: int = Field(ge=0)
    quality_warnings: list[str] = Field(default_factory=list)
    generated_at: datetime
