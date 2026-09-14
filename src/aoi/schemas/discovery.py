from datetime import datetime

from pydantic import Field, HttpUrl

from .common import AOIBaseModel, SourceType


class DiscoveryStrategy(AOIBaseModel):
    name: str
    purpose: str
    query_templates: list[str] = Field(default_factory=list)


class DiscoveryPlan(AOIBaseModel):
    strategies: list[DiscoveryStrategy]
    created_at: datetime


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
