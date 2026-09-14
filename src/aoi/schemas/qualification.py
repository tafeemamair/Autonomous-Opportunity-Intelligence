from datetime import datetime
from enum import StrEnum

from pydantic import Field

from .common import AOIBaseModel


class QualificationStatus(StrEnum):
    QUALIFIED = "QUALIFIED"
    WATCHLIST = "WATCHLIST"
    DISQUALIFIED = "DISQUALIFIED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ProspectType(StrEnum):
    POTENTIAL_CUSTOMER = "POTENTIAL_CUSTOMER"
    POTENTIAL_PARTNER = "POTENTIAL_PARTNER"
    COMPETITOR_OR_VENDOR = "COMPETITOR_OR_VENDOR"
    UNKNOWN = "UNKNOWN"


class FitLevel(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


class QualificationResult(AOIBaseModel):
    """Structured decision output of the Qualification Agent."""

    candidate_id: str
    company_name: str
    qualification_status: QualificationStatus
    prospect_type: ProspectType
    need_fit: FitLevel
    capability_fit: FitLevel
    commercial_relevance: FitLevel
    timing: FitLevel
    evidence_sufficiency: FitLevel
    reasons: list[str] = Field(default_factory=list)
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    disqualification_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    qualified_at: datetime
