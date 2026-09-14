"""AOI Domain Schemas."""

from .common import (
    AOIBaseModel,
    Evidence,
    Priority,
    RunStatus,
    Source,
    SourceType,
    VerificationStatus,
)
from .discovery import Candidate, DiscoveryPlan, DiscoveryStrategy
from .objective import AOIInput, BusinessObjective, Constraints, OperatorProfile
from .opportunity import AOIReport, Company, DecisionMaker, Opportunity, OpportunityScores
from .qualification import FitLevel, ProspectType, QualificationResult, QualificationStatus
from .research import ResearchResult, ResearchStatus
from .verification import VerificationClaim, VerificationResult

__all__ = [
    "AOIBaseModel",
    "AOIInput",
    "AOIReport",
    "BusinessObjective",
    "Candidate",
    "Company",
    "Constraints",
    "DecisionMaker",
    "DiscoveryPlan",
    "DiscoveryStrategy",
    "Evidence",
    "FitLevel",
    "OperatorProfile",
    "Opportunity",
    "OpportunityScores",
    "Priority",
    "ProspectType",
    "QualificationResult",
    "QualificationStatus",
    "ResearchResult",
    "ResearchStatus",
    "RunStatus",
    "Source",
    "SourceType",
    "VerificationClaim",
    "VerificationResult",
    "VerificationStatus",
]
