from pydantic import Field

from .common import AOIBaseModel


class OperatorProfile(AOIBaseModel):
    capabilities: list[str] = Field(default_factory=list)
    portfolio_projects: list[str] = Field(default_factory=list)
    preferred_services: list[str] = Field(default_factory=list)


class Constraints(AOIBaseModel):
    recency_days: int = Field(default=90, ge=1, le=3650)
    require_evidence: bool = True
    require_human_approval: bool = True


class BusinessObjective(AOIBaseModel):
    description: str = Field(min_length=10)
    target_markets: list[str] = Field(default_factory=list)
    target_company_types: list[str] = Field(default_factory=list)
    target_opportunities: list[str] = Field(default_factory=list)
    minimum_opportunities: int = Field(default=20, ge=1, le=1000)
    maximum_opportunities: int = Field(default=50, ge=1, le=5000)


class AOIInput(AOIBaseModel):
    objective: BusinessObjective
    operator_profile: OperatorProfile = Field(default_factory=OperatorProfile)
    constraints: Constraints = Field(default_factory=Constraints)
