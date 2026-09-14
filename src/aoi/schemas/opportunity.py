from pydantic import Field, HttpUrl

from .common import AOIBaseModel, Evidence, Priority


class Company(AOIBaseModel):
    name: str
    website: HttpUrl | None = None
    location: str | None = None
    industry: str | None = None


class DecisionMaker(AOIBaseModel):
    name: str | None = None
    role: str | None = None
    profile_url: HttpUrl | None = None


class OpportunityScores(AOIBaseModel):
    need_fit: float = Field(ge=0, le=100)
    capability_fit: float = Field(ge=0, le=100)
    evidence_strength: float = Field(ge=0, le=100)
    timing: float = Field(ge=0, le=100)
    commercial_potential: float = Field(ge=0, le=100)
    accessibility: float = Field(ge=0, le=100)
    opportunity_score: float = Field(ge=0, le=100)
    confidence_score: float = Field(ge=0, le=100)


class Opportunity(AOIBaseModel):
    rank: int = Field(ge=1)
    company: Company
    title: str
    problem: str
    potential_solution: str
    why_now: str | None = None
    why_fit: str | None = None
    decision_maker: DecisionMaker = Field(default_factory=DecisionMaker)
    scores: OpportunityScores
    priority: Priority
    evidence: list[Evidence] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    recommended_action: str = "Human review"
    status: str = "NEW"


class ReportSummary(AOIBaseModel):
    candidates_discovered: int = 0
    candidates_after_deduplication: int = 0
    qualified_opportunities: int = 0
    high_priority_opportunities: int = 0


class RunQuality(AOIBaseModel):
    overall_confidence: float = Field(default=0, ge=0, le=100)
    verification_rate: float = Field(default=0, ge=0, le=100)
    source_quality: float = Field(default=0, ge=0, le=100)


class AOIReport(AOIBaseModel):
    run_id: str
    generated_at: str
    objective: dict
    summary: ReportSummary
    opportunities: list[Opportunity] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    run_quality: RunQuality = Field(default_factory=RunQuality)
