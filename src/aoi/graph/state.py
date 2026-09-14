import operator
from typing import Annotated

from pydantic import Field

from ..schemas.common import AOIBaseModel, RunStatus
from ..schemas.discovery import Candidate, DiscoveryPlan
from ..schemas.objective import AOIInput
from ..schemas.opportunity import AOIReport, Opportunity


class AOIState(AOIBaseModel):
    run_id: str
    input: AOIInput
    status: RunStatus = RunStatus.CREATED
    discovery_plan: DiscoveryPlan | None = None
    candidates: Annotated[list[Candidate], operator.add] = Field(default_factory=list)
    opportunities: Annotated[list[Opportunity], operator.add] = Field(default_factory=list)
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)
    warnings: Annotated[list[str], operator.add] = Field(default_factory=list)
    final_report: AOIReport | None = None
