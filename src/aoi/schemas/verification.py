from datetime import datetime

from pydantic import Field

from .common import AOIBaseModel, VerificationStatus


class VerificationClaim(AOIBaseModel):
    """A specific factual claim evaluated by the Verification Agent."""

    claim: str = Field(min_length=1)
    status: VerificationStatus
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    corroborating_evidence_ids: list[str] = Field(default_factory=list)
    contradiction_evidence_ids: list[str] = Field(default_factory=list)
    explanation: str = Field(min_length=1)


class VerificationResult(AOIBaseModel):
    """Structured audit and validation report produced by the Verification Agent."""

    candidate_id: str
    company_name: str
    verification_status: VerificationStatus
    verified_claims: list[VerificationClaim] = Field(default_factory=list)
    partially_verified_claims: list[VerificationClaim] = Field(default_factory=list)
    unverified_claims: list[VerificationClaim] = Field(default_factory=list)
    contradicted_claims: list[VerificationClaim] = Field(default_factory=list)
    verification_warnings: list[str] = Field(default_factory=list)
    evidence_checked: int = 0
    independent_sources: int = 0
    verified_at: datetime
