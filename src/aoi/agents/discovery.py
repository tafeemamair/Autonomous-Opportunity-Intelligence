from datetime import UTC, datetime
from uuid import uuid4

from ..schemas.common import SourceType
from ..schemas.discovery import Candidate, DiscoveryPlan, DiscoveryStrategy
from ..schemas.objective import AOIInput

DEFAULT_STRATEGIES = {
    "hiring_signals": (
        "Find current hiring signals that indicate demand for AI, automation, workflow, agent, or engineering capabilities.",
        ['"{opportunity}" "{market}" hiring', '"{opportunity}" "{market}" jobs'],
    ),
    "product_signals": (
        "Find product launches, AI initiatives, automation announcements, or new technical capabilities that may create implementation needs.",
        ['"{opportunity}" "{market}" launch', '"{opportunity}" "{market}" announcement'],
    ),
    "problem_signals": (
        "Find public evidence of operational bottlenecks, repetitive workflows, scaling problems, or other automation opportunities.",
        ['"{problem}" "{market}" company'],
    ),
    "growth_signals": (
        "Find growth, funding, expansion, and hiring events that may create new technical automation demand.",
        ['"{market}" startup funding "{opportunity}"', '"{market}" company expansion "{opportunity}"'],
    ),
    "technology_signals": (
        "Find companies adopting AI, agents, LLMs, or automation technologies.",
        ['"{technology}" "{market}" company'],
    ),
}


class DiscoveryAgent:
    """Schema-first discovery planner.

    No LLM or search API is used yet. This agent creates a deterministic plan that
    a provider adapter will execute in a later milestone.
    """

    def plan(self, aoi_input: AOIInput) -> DiscoveryPlan:
        objective = aoi_input.objective
        market = (objective.target_markets or ["global"])[0]
        opportunity = (objective.target_opportunities or ["AI automation"])[0]
        strategies = []

        for name, (purpose, templates) in DEFAULT_STRATEGIES.items():
            expanded = [
                template.format(
                    opportunity=opportunity,
                    market=market,
                    problem=objective.description,
                    technology=opportunity,
                )
                for template in templates
            ]
            strategies.append(
                DiscoveryStrategy(name=name, purpose=purpose, query_templates=expanded)
            )

        return DiscoveryPlan(strategies=strategies, created_at=datetime.now(UTC))

    def normalize_candidate(
        self,
        company_name: str,
        discovery_strategy: str,
        discovery_reason: str,
        website: str | None = None,
        location: str | None = None,
        source_urls: list[str] | None = None,
    ) -> Candidate:
        urls = source_urls or []
        return Candidate(
            candidate_id=f"cand_{uuid4().hex[:12]}",
            company_name=company_name,
            website=website,
            location=location,
            discovery_strategy=discovery_strategy,
            discovery_reason=discovery_reason,
            source_urls=urls,
            source_types=[SourceType.OTHER] * len(urls),
            discovered_at=datetime.now(UTC),
        )
