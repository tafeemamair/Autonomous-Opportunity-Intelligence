from aoi.agents.discovery import DiscoveryAgent
from aoi.schemas.objective import AOIInput, BusinessObjective


def test_discovery_plan_is_schema_valid():
    input_data = AOIInput(objective=BusinessObjective(
        description="Find companies with recent AI automation needs.",
        target_markets=["United States"],
        target_opportunities=["AI agents"],
    ))
    plan = DiscoveryAgent().plan(input_data)
    assert len(plan.strategies) == 5
    assert all(strategy.query_templates for strategy in plan.strategies)
    assert "AI agents" in plan.strategies[0].query_templates[0]


def test_candidate_normalization():
    candidate = DiscoveryAgent().normalize_candidate(
        company_name="Example AI",
        discovery_strategy="hiring_signals",
        discovery_reason="Current AI engineering job listing.",
        website="https://example.com",
        location="United States",
        source_urls=["https://example.com/careers"],
    )
    assert candidate.company_name == "Example AI"
    assert candidate.source_urls[0].host == "example.com"
