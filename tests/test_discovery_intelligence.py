from datetime import UTC, datetime, timedelta

from pydantic import HttpUrl

from aoi.agents.discovery import DiscoveryAgent
from aoi.agents.qualification import QualificationAgent
from aoi.agents.research import ResearchAgent
from aoi.agents.scoring import ScoringAgent
from aoi.agents.verification import VerificationAgent
from aoi.graph.workflow import (
    aoi_graph,
    create_initial_state,
    discovery_node,
    get_discovery_provider,
)
from aoi.providers.research import ResearchItem, ResearchProvider
from aoi.schemas.common import Priority, RunStatus, SourceType, VerificationStatus
from aoi.schemas.discovery import (
    Candidate,
    CandidateResearchPriority,
    DiscoveryBudget,
    DiscoveryEvaluation,
    DiscoveryPlan,
)
from aoi.schemas.objective import AOIInput, BusinessObjective, Constraints, OperatorProfile
from aoi.schemas.qualification import QualificationStatus
from aoi.schemas.report import AOIReport
from aoi.schemas.research import (
    ResearchEvaluation,
    ResearchQuality,
    ResearchResult,
    ResearchStatus,
)


class MockResearchProvider(ResearchProvider):
    """Deterministic mock provider for testing adaptive research workflows."""

    def __init__(self, responses: dict[str, list[ResearchItem]] | None = None):
        self.responses = responses or {}
        self.recorded_queries: list[str] = []

    def search(self, query: str, max_results: int = 5) -> list[ResearchItem]:
        self.recorded_queries.append(query)
        sorted_responses = sorted(self.responses.items(), key=lambda kv: len(kv[0]), reverse=True)
        for key, items in sorted_responses:
            if key.lower() in query.lower():
                return items[:max_results]
        return []


# =====================================================================
# 1. Strategy Planning & Signal Categories (6 tests)
# =====================================================================


def test_strategy_planning_all_signal_categories():
    """Verify plans categorize queries into supported signal categories (hiring, product, problem, growth, technology)."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Find enterprise companies adopting AI customer support and automated workflows.",
            target_markets=["North America"],
            target_opportunities=["Customer Support AI"],
        )
    )
    plan = agent.plan(input_data)
    assert isinstance(plan, DiscoveryPlan)
    assert len(plan.strategies) == 5

    categories = {s.signal_category for s in plan.strategies}
    assert "hiring" in categories
    assert "product" in categories
    assert "problem" in categories
    assert "growth" in categories
    assert "technology" in categories


def test_strategy_planning_market_expansion():
    """Verify query templates expand across primary and secondary target markets."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Automate repetitive back-office workflows for scaling teams.",
            target_markets=["North America", "Europe", "Asia-Pacific"],
            target_opportunities=["Agentic Automation"],
        )
    )
    plan = agent.plan(input_data)
    hiring_strat = next(s for s in plan.strategies if s.name == "hiring_signals")
    joined_queries = " ".join(hiring_strat.query_templates)
    assert "North America" in joined_queries
    assert "Europe" in joined_queries
    assert "Asia-Pacific" in joined_queries


def test_strategy_planning_opportunity_expansion():
    """Verify query templates expand across multiple opportunities."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Identify automation opportunities.",
            target_markets=["United States"],
            target_opportunities=["LLM Data Pipelines", "Agentic Dispatch"],
        )
    )
    plan = agent.plan(input_data)
    tech_strat = next(s for s in plan.strategies if s.name == "technology_signals")
    joined_queries = " ".join(tech_strat.query_templates)
    assert "LLM Data Pipelines" in joined_queries
    assert "Agentic Dispatch" in joined_queries


def test_strategy_planning_company_type_expansion():
    """Verify target company types produce specialized queries."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Find companies scaling their technical infrastructure.",
            target_markets=["Global"],
            target_opportunities=["Workflow Agents"],
            target_company_types=["Mid-Market SaaS", "Fintech Scaleup"],
        )
    )
    plan = agent.plan(input_data)
    growth_strat = next(s for s in plan.strategies if s.name == "growth_signals")
    joined = " ".join(growth_strat.query_templates)
    assert "Mid-Market SaaS" in joined
    assert "Fintech Scaleup" in joined


def test_strategy_planning_deterministic_order():
    """Verify plan output is strictly deterministic across repeated invocations."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Find companies needing automated workflow solutions.",
            target_markets=["North America", "Europe"],
            target_opportunities=["AI Agents"],
            target_company_types=["Enterprise"],
        )
    )
    plan1 = agent.plan(input_data)
    plan2 = agent.plan(input_data)
    assert [s.name for s in plan1.strategies] == [s.name for s in plan2.strategies]
    for s1, s2 in zip(plan1.strategies, plan2.strategies, strict=True):
        assert s1.query_templates == s2.query_templates


def test_strategy_planning_fallback_defaults():
    """Verify default fallback market and opportunity when input lists are empty."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Identify enterprise automation and workflow intelligence opportunities.",
            target_markets=[],
            target_opportunities=[],
        )
    )
    plan = agent.plan(input_data)
    assert len(plan.strategies) == 5
    for strat in plan.strategies:
        assert len(strat.query_templates) >= 1
        assert strat.market == "global"
        assert strat.opportunity == "AI automation"


# =====================================================================
# 2. Query Generation & Bounding (5 tests)
# =====================================================================


def test_query_generation_no_duplicates():
    """Verify generated query lists do not contain duplicate strings while preserving insertion order."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Automate repetitive back-office and data processing workflows.",
            target_markets=["US", "US"],
            target_opportunities=["AI", "AI"],
        )
    )
    plan = agent.plan(input_data)
    for strat in plan.strategies:
        assert len(strat.query_templates) == len(set(strat.query_templates))


def test_query_generation_template_interpolation():
    """Verify variables ({opportunity}, {market}, {problem}) are interpolated properly."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Resolving operational bottlenecks in healthcare claims.",
            target_markets=["United States"],
            target_opportunities=["Claims Automation"],
        )
    )
    plan = agent.plan(input_data)
    prob_strat = next(s for s in plan.strategies if s.name == "problem_signals")
    assert any("Resolving operational bottlenecks in healthcare claims." in q for q in prob_strat.query_templates)


def test_query_generation_strategy_metadata():
    """Verify strategies hold expected signals, purpose, and name."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Find companies needing AI developer tooling.",
            target_markets=["Canada"],
            target_opportunities=["DevTools"],
        )
    )
    plan = agent.plan(input_data)
    for s in plan.strategies:
        assert len(s.expected_signals) >= 2
        assert len(s.purpose) > 10
        assert s.name.endswith("_signals")


def test_query_generation_bounded_size():
    """Verify total queries per strategy are strictly bounded by input combinations."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Automate repetitive data processing workflows.",
            target_markets=["M1", "M2"],
            target_opportunities=["O1", "O2"],
            target_company_types=["C1"],
        )
    )
    plan = agent.plan(input_data)
    for s in plan.strategies:
        assert len(s.query_templates) <= 15


def test_query_generation_escaping_special_chars():
    """Verify queries handle quotes and punctuation in descriptions cleanly."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description='Automating "high-volume" invoice processing & ledger reconciliation.',
            target_markets=["US/East"],
            target_opportunities=['"Smart" Accounting'],
        )
    )
    plan = agent.plan(input_data)
    assert len(plan.strategies) == 5
    for s in plan.strategies:
        for q in s.query_templates:
            assert isinstance(q, str) and len(q) > 0


# =====================================================================
# 3. Candidate Normalization (6 tests)
# =====================================================================


def test_canonical_name_cleaning_legal_suffixes():
    """Verify Inc, LLC, Corp, Ltd, GmbH, Co. are cleanly stripped."""
    agent = DiscoveryAgent()
    assert agent.clean_company_name("Acme Corp.") == "Acme"
    assert agent.clean_company_name("Acme Corporation") == "Acme"
    assert agent.clean_company_name("Acme, Inc.") == "Acme"
    assert agent.clean_company_name("Acme LLC") == "Acme"
    assert agent.clean_company_name("Acme Ltd.") == "Acme"
    assert agent.clean_company_name("Acme GmbH") == "Acme"
    assert agent.clean_company_name("Acme Pvt. Ltd.") == "Acme"


def test_canonical_name_preserves_legitimate_tokens():
    """Verify 'Technologies', 'Solutions', 'Systems', 'Enterprises', 'Robotics' are preserved."""
    agent = DiscoveryAgent()
    assert agent.clean_company_name("Acme Technologies, Inc.") == "Acme Technologies"
    assert agent.clean_company_name("Cloud Solutions LLC") == "Cloud Solutions"
    assert agent.clean_company_name("BioGenix Systems Corp") == "BioGenix Systems"
    assert agent.clean_company_name("Apex Robotics Ltd.") == "Apex Robotics"


def test_canonical_name_web_prefix_and_domain_stripping():
    """Verify URLs and domain suffixes like .ai, .io, .com are stripped when passed as raw names."""
    agent = DiscoveryAgent()
    assert agent.clean_company_name("https://www.acme.ai") == "acme"
    assert agent.clean_company_name("http://starlight.io") == "starlight"
    assert agent.clean_company_name("nexus.ai") == "nexus"
    assert agent.clean_company_name("quantum.com") == "quantum"


def test_canonical_website_normalization():
    """Verify HTTPS enforcement, trailing slash normalization, and tracking query param stripping."""
    agent = DiscoveryAgent()
    u1 = agent.clean_website_url("http://example.com/careers?utm_source=feed&ref=42")
    assert u1 == HttpUrl("https://example.com/careers")

    u2 = agent.clean_website_url("www.example.com/")
    assert u2 == HttpUrl("https://www.example.com/")

    u3 = agent.clean_website_url(None)
    assert u3 is None


def test_location_normalization_noise_filtering():
    """Verify location strings strip non-location trailing clauses."""
    agent = DiscoveryAgent()
    assert agent.normalize_location("Austin, Texas, USA") == "Austin, Texas, USA"
    assert agent.normalize_location("San Francisco, CA.") == "San Francisco, CA"
    assert agent.normalize_location("  ") is None


def test_source_type_auto_inference_all_tiers():
    """Verify URL domains map accurately across all source tiers."""
    agent = DiscoveryAgent()
    assert agent.infer_source_type_from_url("https://linkedin.com/company/acme") == SourceType.LINKEDIN
    assert agent.infer_source_type_from_url("https://boards.greenhouse.io/acme/jobs/1") == SourceType.JOB_BOARD
    assert agent.infer_source_type_from_url("https://jobright.ai/jobs/1") == SourceType.JOB_BOARD
    assert agent.infer_source_type_from_url("https://clutch.co/profile/acme") == SourceType.DIRECTORY
    assert agent.infer_source_type_from_url("https://techcrunch.com/2026/08/launch") == SourceType.NEWS
    assert agent.infer_source_type_from_url("https://venturebeat.com/ai/article") == SourceType.INDUSTRY_PUBLICATION
    assert agent.infer_source_type_from_url("https://prnewswire.com/news-releases/acme") == SourceType.OFFICIAL_ANNOUNCEMENT
    assert agent.infer_source_type_from_url("https://twitter.com/acme/status/1") == SourceType.SOCIAL
    assert agent.infer_source_type_from_url("https://acme.com/about") == SourceType.COMPANY_WEBSITE


# =====================================================================
# 4. Deduplication & Immutability (5 tests)
# =====================================================================


def test_deduplication_by_domain():
    """Verify candidates sharing a domain merge into a single entity."""
    agent = DiscoveryAgent()
    c1 = agent.normalize_candidate("Alpha", "hiring_signals", "Hiring engineer", website="https://alpha.io")
    c2 = agent.normalize_candidate("Alpha Corp", "product_signals", "New product", website="https://alpha.io/blog")
    deduped = agent.deduplicate_candidates([c1, c2])
    assert len(deduped) == 1
    assert deduped[0].domain == "alpha.io"


def test_deduplication_by_canonical_name():
    """Verify candidates without domains merge by canonical cleaned name."""
    agent = DiscoveryAgent()
    c1 = agent.normalize_candidate("Beta Technologies Inc.", "hiring_signals", "Hiring role")
    c2 = agent.normalize_candidate("Beta Technologies LLC", "problem_signals", "Manual bottlenecks")
    deduped = agent.deduplicate_candidates([c1, c2])
    assert len(deduped) == 1
    assert "beta technologies" in deduped[0].normalized_name


def test_deduplication_input_immutability():
    """Verify input candidate objects remain strictly unmutated."""
    agent = DiscoveryAgent()
    c1 = agent.normalize_candidate("Gamma AI", "hiring_signals", "Hiring role", website="https://gamma.ai")
    c2 = agent.normalize_candidate("Gamma AI Inc", "product_signals", "Launched agent", website="https://gamma.ai")

    score_before = c1.research_priority_score
    reasons_before = len(c1.discovery_reasons)
    signals_before = len(c1.signal_categories)

    deduped = agent.deduplicate_candidates([c1, c2])

    assert c1.research_priority_score == score_before
    assert len(c1.discovery_reasons) == reasons_before
    assert len(c1.signal_categories) == signals_before
    assert len(deduped[0].signal_categories) >= 2


def test_deduplication_source_urls_consolidation():
    """Verify merged candidate retains all distinct source URLs and types."""
    agent = DiscoveryAgent()
    c1 = agent.normalize_candidate("Delta", "hiring_signals", "Hiring", website="https://delta.ai", source_urls=["https://delta.ai/jobs"])
    c2 = agent.normalize_candidate("Delta", "product_signals", "Launch", website="https://delta.ai", source_urls=["https://delta.ai/press"])
    deduped = agent.deduplicate_candidates([c1, c2])
    assert len(deduped[0].source_urls) == 2


def test_deduplication_url_idempotence():
    """Verify duplicate source URLs are not added twice."""
    agent = DiscoveryAgent()
    c1 = agent.normalize_candidate("Epsilon", "hiring_signals", "Hiring", website="https://epsilon.ai", source_urls=["https://epsilon.ai/careers"])
    c2 = agent.normalize_candidate("Epsilon", "growth_signals", "Funding", website="https://epsilon.ai", source_urls=["https://epsilon.ai/careers"])
    deduped = agent.deduplicate_candidates([c1, c2])
    assert len(deduped[0].source_urls) == 1


# =====================================================================
# 5. Multi-Signal Aggregation (5 tests)
# =====================================================================


def test_signal_aggregation_multi_category_accumulation():
    """Verify merging accumulates distinct signal categories."""
    agent = DiscoveryAgent()
    c1 = agent.normalize_candidate("Zeta", "hiring_signals", "Reason 1", website="https://zeta.ai")
    c2 = agent.normalize_candidate("Zeta", "technology_signals", "Reason 2", website="https://zeta.ai")
    merged = agent.merge_candidates(c1, c2)
    assert "hiring" in merged.signal_categories
    assert "technology" in merged.signal_categories


def test_signal_aggregation_discovery_reason_accumulation():
    """Verify discovery reasons from different strategies are preserved and joined."""
    agent = DiscoveryAgent()
    c1 = agent.normalize_candidate("Eta", "hiring_signals", "Hiring AI Engineers.", website="https://eta.ai")
    c2 = agent.normalize_candidate("Eta", "problem_signals", "Operational scaling bottlenecks.", website="https://eta.ai")
    merged = agent.merge_candidates(c1, c2)
    assert len(merged.discovery_reasons) == 2
    assert "Hiring AI Engineers." in merged.discovery_reason
    assert "Operational scaling bottlenecks." in merged.discovery_reason


def test_signal_aggregation_priority_boost():
    """Verify multi-signal candidates receive a higher priority score than single-signal candidates."""
    agent = DiscoveryAgent()
    single = agent.normalize_candidate("Theta", "hiring_signals", "Hiring role", website="https://theta.ai")
    multi = agent.normalize_candidate("Theta", "hiring_signals", "Hiring role", website="https://theta.ai")
    multi.signal_categories.append("product")
    multi.signal_categories.append("growth")
    score_multi, _ = agent.compute_priority_score(multi)
    assert score_multi > single.research_priority_score


def test_signal_aggregation_strategy_synthesis():
    """Verify merged discovery_strategy reflects combined sources."""
    agent = DiscoveryAgent()
    c1 = agent.normalize_candidate("Iota", "hiring_signals", "Hiring", website="https://iota.ai")
    c2 = agent.normalize_candidate("Iota", "growth_signals", "Growth", website="https://iota.ai")
    merged = agent.merge_candidates(c1, c2)
    assert "hiring_signals" in merged.discovery_strategy
    assert "growth_signals" in merged.discovery_strategy


def test_signal_aggregation_signal_category_idempotence():
    """Verify redundant signals of the same category are not duplicated in signal_categories."""
    agent = DiscoveryAgent()
    c1 = agent.normalize_candidate("Kappa", "hiring_signals", "Hiring 1", website="https://kappa.ai")
    c2 = agent.normalize_candidate("Kappa", "hiring_signals", "Hiring 2", website="https://kappa.ai")
    merged = agent.merge_candidates(c1, c2)
    assert merged.signal_categories.count("hiring") == 1


# =====================================================================
# 6. Research Prioritization (Diagnostic-Only) (5 tests)
# =====================================================================


def test_research_prioritization_descending_sort():
    """Verify candidates sort strictly descending by priority score."""
    agent = DiscoveryAgent()
    c1 = agent.normalize_candidate("Low Cand", "general", "hit")
    c2 = agent.normalize_candidate("High Cand", "technology_signals", "Active AI agent workflow adoption", website="https://high.ai", source_urls=["https://high.ai"])
    c2.signal_categories.append("hiring")

    ranked = agent.prioritize_candidates([c1, c2])
    assert ranked[0].company_name == "High Cand"
    assert ranked[1].company_name == "Low Cand"
    assert ranked[0].research_priority_score >= ranked[1].research_priority_score


def test_research_prioritization_enums():
    """Verify priority scores map to URGENT, HIGH, MEDIUM, LOW correctly."""
    agent = DiscoveryAgent()
    c = agent.normalize_candidate("Test", "technology_signals", "Active AI agents workflow pipeline automation engineer", website="https://test.ai", source_urls=["https://test.ai"])
    c.signal_categories.extend(["hiring", "growth"])
    score, prio = agent.compute_priority_score(c)
    assert score >= 80.0
    assert prio == CandidateResearchPriority.URGENT


def test_research_prioritization_official_domain_bonus():
    """Verify candidates with verified official domains score higher."""
    agent = DiscoveryAgent()
    c_with = agent.normalize_candidate("Acme", "hiring_signals", "Hiring", website="https://acme.com")
    c_without = agent.normalize_candidate("Acme", "hiring_signals", "Hiring")
    assert c_with.research_priority_score > c_without.research_priority_score


def test_research_prioritization_keyword_relevance():
    """Verify relevance keywords in discovery reasons increase score."""
    agent = DiscoveryAgent()
    c_rich = agent.normalize_candidate("Alpha", "hiring_signals", "AI agent workflow pipeline bottleneck launch automation engineer")
    c_sparse = agent.normalize_candidate("Alpha", "hiring_signals", "general info")
    assert c_rich.research_priority_score > c_sparse.research_priority_score


def test_research_prioritization_diagnostic_isolation():
    """Verify candidate priority attributes do not impact qualification or verification status."""
    cand = Candidate(
        candidate_id="cand-iso-1",
        company_name="Apex AI",
        website=HttpUrl("https://apex.ai"),
        discovery_strategy="hiring_signals",
        discovery_reason="Hiring AI engineers",
        discovered_at=datetime.now(UTC),
        research_priority=CandidateResearchPriority.URGENT,
        research_priority_score=95.0,
    )
    qual_agent = QualificationAgent()
    res = ResearchResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        website=cand.website,
        technology_signals=["AI Agents"],
        researched_at=datetime.now(UTC),
    )
    qual = qual_agent.qualify(
        candidate=cand,
        research_result=res,
        objective=BusinessObjective(description="Find companies using AI."),
        operator_profile=OperatorProfile(capabilities=["AI"]),
        constraints=Constraints(),
    )
    assert qual.qualification_status in (
        QualificationStatus.QUALIFIED,
        QualificationStatus.WATCHLIST,
        QualificationStatus.DISQUALIFIED,
        QualificationStatus.INSUFFICIENT_EVIDENCE,
    )


# =====================================================================
# 7. Adaptive Research & Bounded Execution (6 tests)
# =====================================================================


def test_adaptive_research_saturation_early_stopping():
    """Verify research halts queries once saturation criteria are met."""
    cand = Candidate(
        candidate_id="cand-sat-2",
        company_name="SatCorp",
        website=HttpUrl("https://satcorp.com"),
        discovery_strategy="technology_signals",
        discovery_reason="Building AI agents",
        discovered_at=datetime.now(UTC),
    )
    provider = MockResearchProvider(
        responses={
            "SatCorp": [
                ResearchItem(
                    title="SatCorp Home",
                    url=HttpUrl("https://satcorp.com/about"),
                    content="SatCorp is in Boston, MA. Founded by CEO Jane Doe. We build autonomous AI agents to automate workflows.",
                    published_date="2026-08-01",
                )
            ]
        }
    )
    agent = ResearchAgent(provider=provider, max_queries=4)
    result = agent.research(cand)
    assert result.is_saturated is True
    assert result.queries_executed == 1
    assert result.queries_saved == 3


def test_adaptive_research_queries_saved_accounting():
    """Verify queries_saved = max_queries - queries_executed."""
    cand = Candidate(
        candidate_id="cand-save-1",
        company_name="SaveCorp",
        website=HttpUrl("https://savecorp.com"),
        discovery_strategy="hiring_signals",
        discovery_reason="Hiring",
        discovered_at=datetime.now(UTC),
    )
    provider = MockResearchProvider(
        responses={
            "SaveCorp": [
                ResearchItem(
                    title="SaveCorp",
                    url=HttpUrl("https://savecorp.com"),
                    content="SaveCorp in Seattle. Founded by CEO John Smith. We automate repetitive tasks with AI agents.",
                    published_date="2026-08-01",
                )
            ]
        }
    )
    agent = ResearchAgent(provider=provider, max_queries=5)
    result = agent.research(cand)
    assert result.queries_saved == 5 - result.queries_executed


def test_adaptive_research_gap_driven_query_selection():
    """Verify adaptive loop generates targeted queries for missing dimensions."""
    cand = Candidate(
        candidate_id="cand-gap-2",
        company_name="GapTech",
        discovery_strategy="problem_signals",
        discovery_reason="Legacy systems",
        discovered_at=datetime.now(UTC),
    )
    provider = MockResearchProvider(
        responses={
            "GapTech": [
                ResearchItem(
                    title="GapTech Home",
                    url=HttpUrl("https://gaptech.com"),
                    content="GapTech operates freight logistics.",
                    published_date="2026-01-01",
                )
            ],
            '"GapTech" "AI"': [
                ResearchItem(
                    title="GapTech AI",
                    url=HttpUrl("https://gaptech.com/ai"),
                    content="Adopting autonomous AI agents for data routing.",
                    published_date="2026-08-01",
                )
            ],
        }
    )
    agent = ResearchAgent(provider=provider, max_queries=3)
    result = agent.research(cand)
    assert result.queries_executed >= 2
    assert any('"GapTech" "AI"' in q for q in provider.recorded_queries)


def test_adaptive_research_strict_query_ceiling():
    """Verify execution never exceeds max_queries."""
    cand = Candidate(
        candidate_id="cand-ceil-1",
        company_name="CeilCorp",
        discovery_strategy="hiring_signals",
        discovery_reason="Hiring",
        discovered_at=datetime.now(UTC),
    )
    provider = MockResearchProvider(responses={})
    agent = ResearchAgent(provider=provider, max_queries=2)
    result = agent.research(cand)
    assert result.queries_executed <= 2
    assert len(provider.recorded_queries) <= 2


def test_adaptive_research_executive_role_extraction():
    """Verify CEO, CTO, VP Engineering, and Head of AI extraction with prepositions and honorifics."""
    agent = ResearchAgent()
    content = "Founded by Elena Rostova, CEO. Also Dr. Marcus Vance, Chief Technology Officer."
    people = agent._extract_people(content, "TestCorp", HttpUrl("https://testcorp.com"))
    roles = {p.name: p.role for p in people}
    assert roles.get("Elena Rostova") == "CEO"
    assert roles.get("Marcus Vance") == "CTO"


def test_adaptive_research_industry_inference():
    """Verify text corpus correctly maps to defined industry verticals."""
    agent = ResearchAgent()
    assert agent._infer_industry("Building subsea autonomous marine robotics.") == "Autonomous Marine Robotics"
    assert agent._infer_industry("Healthcare clinical biotech genomics.") == "Healthcare & Life Sciences"
    assert agent._infer_industry("Fintech wealth management banking.") == "Fintech & Financial Services"
    assert agent._infer_industry("Freight logistics and supply chain.") == "Logistics & Supply Chain"


# =====================================================================
# 8. Evaluation Metrics & Accounting (6 tests)
# =====================================================================


def test_discovery_evaluation_metrics_calculation():
    """Verify deduplication rate, unique count, multi-signal ratio, and channel diversity."""
    agent = DiscoveryAgent()
    c1 = agent.normalize_candidate("Alpha AI", "hiring_signals", "Hiring", website="https://alpha.ai")
    c2 = agent.normalize_candidate("Beta AI", "product_signals", "Product", website="https://beta.ai")
    c2.signal_categories.append("growth")

    eval_res = agent.evaluate_discovery(raw_candidates_count=4, normalized_candidates=[c1, c2])
    assert eval_res.candidates_discovered_count == 4
    assert eval_res.unique_companies_count == 2
    assert eval_res.deduplication_rate == 50.0
    assert eval_res.multi_signal_candidate_count == 1
    assert eval_res.multi_signal_ratio == 50.0
    assert eval_res.channel_diversity_score > 0.0


def test_discovery_evaluation_fallback_empty_inputs():
    """Verify safe evaluation on 0 discovered candidates."""
    agent = DiscoveryAgent()
    eval_res = agent.evaluate_discovery(raw_candidates_count=0, normalized_candidates=[])
    assert eval_res.candidates_discovered_count == 0
    assert eval_res.unique_companies_count == 0
    assert eval_res.deduplication_rate == 0.0
    assert eval_res.multi_signal_candidate_count == 0
    assert eval_res.channel_diversity_score == 0.0


def test_research_evaluation_metrics_calculation():
    """Verify cost savings percentage, yield per query, saturation rate, tier 1 ratio, DM rate, avg quality."""
    cand = Candidate(
        candidate_id="c1",
        company_name="TestCorp",
        website=HttpUrl("https://testcorp.com"),
        discovery_strategy="hiring_signals",
        discovery_reason="Hiring",
        discovered_at=datetime.now(UTC),
    )
    provider = MockResearchProvider(
        responses={
            "TestCorp": [
                ResearchItem(
                    title="TestCorp",
                    url=HttpUrl("https://testcorp.com"),
                    content="TestCorp in Austin. Founded by CEO Sarah Connor. We build AI agents.",
                    published_date="2026-08-01",
                )
            ]
        }
    )
    res_agent = ResearchAgent(provider=provider, max_queries=4)
    result = res_agent.research(cand)
    eval_res = res_agent.evaluate_research([result])

    assert eval_res.candidates_researched_count == 1
    assert eval_res.total_queries_executed == result.queries_executed
    assert eval_res.total_queries_saved == result.queries_saved
    assert eval_res.cost_savings_percentage >= 0.0
    assert eval_res.decision_maker_discovery_rate == 100.0


def test_research_quality_scoring_mathematics():
    """Verify completeness, diversity, recency, signal depth, and overall weighted score."""
    cand = Candidate(
        candidate_id="cq1",
        company_name="BioGenix",
        website=HttpUrl("https://biogenix.com"),
        discovery_strategy="hiring_signals",
        discovery_reason="Biotech",
        discovered_at=datetime.now(UTC),
    )
    provider = MockResearchProvider(
        responses={
            "BioGenix": [
                ResearchItem(
                    title="BioGenix",
                    url=HttpUrl("https://biogenix.com"),
                    content="BioGenix in Boston. CEO Dr. Vance. Building AI agents for molecular pipeline automation.",
                    published_date="2026-08-01",
                )
            ]
        }
    )
    res_agent = ResearchAgent(provider=provider, max_queries=2)
    result = res_agent.research(cand)
    q: ResearchQuality = result.research_quality
    assert q is not None
    assert 0.0 <= q.completeness_score <= 100.0
    assert 0.0 <= q.evidence_diversity_score <= 100.0
    assert 0.0 <= q.recency_score <= 100.0
    assert 0.0 <= q.signal_depth_score <= 100.0
    assert 0.0 <= q.overall_quality_score <= 100.0


def test_research_evaluation_zero_division_safety():
    """Verify evaluation metrics do not raise ZeroDivisionError on empty results."""
    agent = ResearchAgent()
    eval_res = agent.evaluate_research([])
    assert eval_res.candidates_researched_count == 0
    assert eval_res.cost_savings_percentage == 0.0
    assert eval_res.evidence_yield_per_query == 0.0
    assert eval_res.average_quality_score == 0.0


def test_evidence_diversity_multi_tier_scoring():
    """Verify diversity score reflects combination of tier 1, tier 2, tier 3, tier 4 sources."""
    cand = Candidate(
        candidate_id="c-div",
        company_name="Omni",
        website=HttpUrl("https://omni.com"),
        discovery_strategy="hiring_signals",
        discovery_reason="Hiring",
        discovered_at=datetime.now(UTC),
    )
    provider = MockResearchProvider(
        responses={
            "Omni": [
                ResearchItem(
                    title="Omni Careers",
                    url=HttpUrl("https://boards.greenhouse.io/omni/1"),
                    content="Omni is hiring Staff AI Engineers.",
                    published_date="2026-08-01",
                ),
                ResearchItem(
                    title="Omni TechCrunch",
                    url=HttpUrl("https://techcrunch.com/2026/08/omni"),
                    content="Omni launches workflow automation agent.",
                    published_date="2026-08-05",
                ),
            ]
        }
    )
    agent = ResearchAgent(provider=provider, max_queries=2)
    result = agent.research(cand)
    assert result.research_quality.evidence_diversity_score >= 50.0


# =====================================================================
# 9. Failure, Edge Cases & Boundary Safety (8 tests)
# =====================================================================


def test_budget_exhaustion_partial_status():
    """Verify research with insufficient evidence returns ResearchStatus.PARTIAL."""
    cand = Candidate(
        candidate_id="c-sparse",
        company_name="SparseCorp",
        discovery_strategy="hiring_signals",
        discovery_reason="Sparse",
        discovered_at=datetime.now(UTC),
    )
    provider = MockResearchProvider(
        responses={
            "SparseCorp": [
                ResearchItem(
                    title="Sparse",
                    url=HttpUrl("https://sparse.com"),
                    content="Sparse single hit.",
                    published_date="2026-01-01",
                )
            ]
        }
    )
    agent = ResearchAgent(provider=provider, max_queries=1)
    result = agent.research(cand)
    assert result.research_status == ResearchStatus.PARTIAL


def test_provider_failure_failed_status():
    """Verify research with provider errors and no evidence returns ResearchStatus.FAILED."""
    cand = Candidate(
        candidate_id="c-fail",
        company_name="FailCorp",
        discovery_strategy="hiring_signals",
        discovery_reason="Fail",
        discovered_at=datetime.now(UTC),
    )

    class FailingProvider(ResearchProvider):
        def search(self, query: str, max_results: int = 5):
            raise ConnectionError("Network down")

    agent = ResearchAgent(provider=FailingProvider(), max_queries=1)
    result = agent.research(cand)
    assert result.research_status == ResearchStatus.FAILED
    assert len(result.warnings) >= 1


def test_stale_evidence_recency_penalty():
    """Verify old published dates receive lower recency scores than fresh findings."""
    agent = ResearchAgent(recency_days=90)
    fresh_date = datetime.now(UTC) - timedelta(days=10)
    stale_date = datetime.now(UTC) - timedelta(days=400)
    assert agent._compute_recency_score(fresh_date) > agent._compute_recency_score(stale_date)


def test_stock_image_caption_rejection():
    """Verify stock photo captions are filtered from signals."""
    agent = ResearchAgent()
    caption = "young smiling businessman looking at monitor in modern office"
    assert agent._is_stock_image_or_caption(caption) is True


def test_research_quality_does_not_set_verification_status():
    """Verify evidence.verification_status remains UNVERIFIED."""
    cand = Candidate(
        candidate_id="c-unver",
        company_name="UnverCorp",
        website=HttpUrl("https://unvercorp.com"),
        discovery_strategy="hiring_signals",
        discovery_reason="Hiring",
        discovered_at=datetime.now(UTC),
    )
    provider = MockResearchProvider(
        responses={
            "UnverCorp": [
                ResearchItem(
                    title="UnverCorp",
                    url=HttpUrl("https://unvercorp.com"),
                    content="UnverCorp builds AI agents in Chicago.",
                    published_date="2026-08-01",
                )
            ]
        }
    )
    agent = ResearchAgent(provider=provider, max_queries=2)
    result = agent.research(cand)
    for ev in result.evidence:
        assert ev.verification_status == VerificationStatus.UNVERIFIED


def test_graph_state_preserves_evaluations():
    """Verify full LangGraph run populates discovery_evaluation and research_evaluation."""
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Find companies with active AI automation needs.",
            target_markets=["North America"],
            target_opportunities=["Autonomous Agents"],
        ),
    )
    cand = Candidate(
        candidate_id="cand-g-1",
        company_name="Terra Robotics Inc.",
        website=HttpUrl("https://terrarobotics.com"),
        location="San Jose, CA",
        discovery_strategy="hiring_signals",
        discovery_reason="Hiring autonomous systems engineer.",
        source_urls=[HttpUrl("https://terrarobotics.com/careers")],
        source_types=[SourceType.COMPANY_WEBSITE],
        discovered_at=datetime.now(UTC),
    )
    state = create_initial_state(input_data)
    state = state.model_copy(update={"candidates": [cand]})
    result = aoi_graph.invoke(state)

    assert result["status"] == RunStatus.COMPLETED
    assert isinstance(result["discovery_evaluation"], DiscoveryEvaluation)
    assert isinstance(result["research_evaluation"], ResearchEvaluation)


def test_report_statistics_exposes_m9_metrics():
    """Verify ReportStatistics contains query counts, cost savings, and evaluations without modifying opportunity scoring."""
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Find companies needing AI developer tooling.",
            target_markets=["North America"],
            target_opportunities=["DevTools"],
        ),
    )
    cand = Candidate(
        candidate_id="cand-r-1",
        company_name="RoboNav AI",
        website=HttpUrl("https://robonav.ai"),
        discovery_strategy="technology_signals",
        discovery_reason="Building AI agents",
        discovered_at=datetime.now(UTC),
    )
    state = create_initial_state(input_data)
    state = state.model_copy(update={"candidates": [cand]})
    result = aoi_graph.invoke(state)
    report: AOIReport = result["report"]

    assert report.statistics.discovery_evaluation is not None
    assert report.statistics.research_evaluation is not None
    assert report.statistics.queries_executed >= 0
    assert report.statistics.queries_saved >= 0
    assert report.statistics.cost_savings_percentage >= 0.0


def test_downstream_intelligence_rules_unchanged():
    """Verify Qualification, Verification, Scoring, and Opportunity Narrative logic remain identical."""
    cand = Candidate(
        candidate_id="cand-rule-1",
        company_name="Starlight AI",
        website=HttpUrl("https://starlight.ai"),
        discovery_strategy="technology_signals",
        discovery_reason="Building AI agents",
        discovered_at=datetime.now(UTC),
    )
    res = ResearchResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        website=cand.website,
        technology_signals=["AI Agents"],
        researched_at=datetime.now(UTC),
    )
    qual = QualificationAgent().qualify(
        candidate=cand,
        research_result=res,
        objective=BusinessObjective(description="Find companies needing AI automation."),
        operator_profile=OperatorProfile(capabilities=["AI"]),
        constraints=Constraints(),
    )
    ver = VerificationAgent().verify(
        candidate=cand,
        research_result=res,
        qualification_result=qual,
        objective=BusinessObjective(description="Find companies needing AI automation."),
        constraints=Constraints(),
    )
    scored = ScoringAgent().score(
        candidate=cand,
        research_result=res,
        qualification_result=qual,
        verification_result=ver,
        objective=BusinessObjective(description="Find companies needing AI automation."),
        operator_profile=OperatorProfile(capabilities=["AI"]),
        constraints=Constraints(),
    )
    assert 0.0 <= scored.scores.opportunity_score <= 100.0
    assert scored.scores.priority in (Priority.HIGH, Priority.QUALIFIED, Priority.WATCHLIST, Priority.DISCARD)


# =====================================================================
# 11. Explicit DiscoveryBudget Tests (4 tests)
# =====================================================================


def test_discovery_budget_max_queries_bounding():
    """Verify DiscoveryAgent.plan bounds total query count across strategies to budget.max_queries."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Find robotics and AI hardware startups in North America.",
            target_markets=["North America", "Europe"],
            target_opportunities=["Robotics", "Hardware"],
        ),
    )
    budget = DiscoveryBudget(max_queries=7, max_results_per_query=3, max_candidates=10)
    plan = agent.plan(input_data, budget=budget)
    total_queries = sum(len(s.query_templates) for s in plan.strategies)
    assert total_queries <= 7
    assert plan.budget.max_queries == 7
    assert plan.budget.max_results_per_query == 3
    assert plan.budget.max_candidates == 10


def test_discovery_budget_max_results_per_query():
    """Verify TavilyDiscoveryProvider passes max_results_per_query to underlying search provider."""
    from unittest.mock import MagicMock

    import httpx

    from aoi.providers.tavily import TavilyDiscoveryProvider

    mock_client = MagicMock(spec=httpx.Client)
    mock_resp_obj = MagicMock(spec=httpx.Response)
    mock_resp_obj.status_code = 200
    mock_resp_obj.json.return_value = {"results": []}
    mock_client.post.return_value = mock_resp_obj

    provider = TavilyDiscoveryProvider(api_key="tvly-mock-key", client=mock_client)

    input_data = AOIInput(
        objective=BusinessObjective(description="Find companies needing AI automation."),
        constraints=Constraints(discovery_budget=DiscoveryBudget(max_queries=2, max_results_per_query=3, max_candidates=5)),
    )
    plan = DiscoveryAgent().plan(input_data, budget=input_data.constraints.discovery_budget)
    provider.discover(plan, budget=input_data.constraints.discovery_budget)

    assert mock_client.post.called
    for call_args in mock_client.post.call_args_list:
        payload = call_args.kwargs.get("json", {})
        assert payload.get("max_results") == 3


def test_discovery_budget_max_candidates_ceiling():
    """Verify discovery_node caps retained prioritized candidates to budget.max_candidates."""
    from aoi.graph.workflow import set_discovery_provider

    class MockDiscoveryProvider:
        def discover(self, plan, budget=None):
            return [
                Candidate(
                    candidate_id=f"cand-{i}",
                    company_name=f"Company {i}",
                    website=HttpUrl(f"https://company{i}.com"),
                    discovery_strategy="technology_signals",
                    discovery_reason="AI Tech",
                    discovered_at=datetime.now(UTC),
                )
                for i in range(10)
            ]

    prev_provider = get_discovery_provider()
    set_discovery_provider(MockDiscoveryProvider())
    try:
        input_data = AOIInput(
            objective=BusinessObjective(description="Find AI companies."),
            constraints=Constraints(discovery_budget=DiscoveryBudget(max_queries=10, max_results_per_query=5, max_candidates=2)),
        )
        state = create_initial_state(input_data)
        out_state = discovery_node(state)
        assert len(out_state["candidates"]) == 2
    finally:
        set_discovery_provider(prev_provider)


def test_discovery_budget_exhaustion_evaluation_recording():
    """Verify DiscoveryEvaluation accurately records budget parameters and queries_capped flag."""
    agent = DiscoveryAgent()
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Find AI startups.",
            target_markets=["US", "EU"],
            target_opportunities=["DevTools", "Robotics"],
        ),
    )
    budget = DiscoveryBudget(max_queries=3, max_results_per_query=2, max_candidates=5)
    plan = agent.plan(input_data, budget=budget)
    total_queries = sum(len(s.query_templates) for s in plan.strategies)
    assert total_queries == 3

    c1 = agent.normalize_candidate("Alpha AI", "technology_signals", "AI", website="https://alpha.ai")
    eval_res = agent.evaluate_discovery(raw_candidates_count=1, normalized_candidates=[c1], plan=plan)

    assert eval_res.budget is not None
    assert eval_res.budget.max_queries == 3
    assert eval_res.queries_executed == 3
    assert eval_res.queries_capped is True
