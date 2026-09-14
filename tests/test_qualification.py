from datetime import UTC, datetime, timedelta

from pydantic import HttpUrl

from aoi.agents.qualification import QualificationAgent
from aoi.graph.workflow import aoi_graph, create_initial_state
from aoi.schemas.common import Evidence, RunStatus, Source, SourceType, VerificationStatus
from aoi.schemas.discovery import Candidate
from aoi.schemas.objective import AOIInput, BusinessObjective, Constraints, OperatorProfile
from aoi.schemas.qualification import (
    FitLevel,
    ProspectType,
    QualificationStatus,
)
from aoi.schemas.research import ResearchResult, ResearchStatus


def _create_sample_candidate(company_name: str = "Bedrock Ocean Exploration") -> Candidate:
    return Candidate(
        candidate_id="cand-1",
        company_name=company_name,
        website=HttpUrl("https://bedrockocean.com"),
        discovery_strategy="hiring_signal",
        discovery_reason="Found open AI engineering role",
        discovered_at=datetime.now(UTC),
    )


def _create_sample_evidence(
    claim: str,
    source_url: str = "https://bedrockocean.com/careers",
    source_type: SourceType = SourceType.JOB_BOARD,
    published_at: datetime | None = None,
    recency_score: int = 85,
) -> Evidence:
    return Evidence(
        claim=claim,
        source=Source(
            url=HttpUrl(source_url),
            source_type=source_type,
            title="Careers page",
            publisher="bedrockocean.com",
            published_at=published_at,
            accessed_at=datetime.now(UTC),
            reliability=85,
        ),
        evidence_summary=claim[:200],
        verification_status=VerificationStatus.UNVERIFIED,
        recency_score=recency_score,
    )


def test_strong_ai_engineering_hiring_qualifies():
    candidate = _create_sample_candidate("Bedrock Ocean Exploration")
    ev1 = _create_sample_evidence(
        claim="Bedrock Ocean Exploration has an open role for a Staff AI Platform Engineer to build autonomous agent data pipelines."
    )
    ev2 = _create_sample_evidence(
        claim="Bedrock Ocean Exploration official website identified at https://bedrockocean.com/",
        source_type=SourceType.COMPANY_WEBSITE,
    )
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="Bedrock Ocean Exploration",
        website=HttpUrl("https://bedrockocean.com"),
        evidence=[ev1, ev2],
        technology_signals=["autonomous agent data pipelines"],
        business_signals=["hiring Staff AI Platform Engineer"],
        problem_signals=["autonomous agent data pipelines"],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    agent = QualificationAgent()
    qual = agent.qualify(candidate, research)

    assert qual.qualification_status == QualificationStatus.QUALIFIED
    assert qual.prospect_type == ProspectType.POTENTIAL_CUSTOMER
    assert qual.need_fit == FitLevel.HIGH
    assert qual.commercial_relevance == FitLevel.HIGH
    assert ev1.id in qual.supporting_evidence_ids


def test_generic_ai_mention_without_clear_need_watchlists():
    candidate = _create_sample_candidate("General Retail Inc")
    ev1 = _create_sample_evidence(
        claim="General Retail mentions AI trends in industry blog.",
        source_type=SourceType.NEWS,
    )
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="General Retail Inc",
        evidence=[ev1],
        technology_signals=["AI trends"],
        business_signals=[],
        problem_signals=[],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    agent = QualificationAgent()
    qual = agent.qualify(candidate, research)

    assert qual.qualification_status == QualificationStatus.WATCHLIST
    assert qual.need_fit == FitLevel.LOW


def test_obvious_ai_agency_vendor_disqualifies():
    candidate = _create_sample_candidate("Intuz")
    ev1 = _create_sample_evidence(
        claim="Intuz is an AI-first development company providing custom AI solutions and AI agent development services for clients.",
        source_type=SourceType.COMPANY_WEBSITE,
    )
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="Intuz",
        website=HttpUrl("https://www.intuz.com"),
        evidence=[ev1],
        technology_signals=["Production AI Agents & Agentic Systems"],
        business_signals=["AI Agent Development Services"],
        problem_signals=["custom AI solutions"],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    agent = QualificationAgent()
    qual = agent.qualify(candidate, research)

    assert qual.prospect_type == ProspectType.COMPETITOR_OR_VENDOR
    assert qual.qualification_status == QualificationStatus.DISQUALIFIED
    assert any("competitor" in r.lower() or "vendor" in r.lower() for r in qual.disqualification_reasons)
    assert ev1.id in qual.supporting_evidence_ids


def test_missing_or_weak_evidence_insufficient():
    candidate = _create_sample_candidate("Ghost Corp")
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="Ghost Corp",
        evidence=[],
        technology_signals=[],
        business_signals=[],
        problem_signals=[],
        research_status=ResearchStatus.FAILED,
        researched_at=datetime.now(UTC),
    )

    agent = QualificationAgent()
    qual = agent.qualify(candidate, research)

    assert qual.qualification_status == QualificationStatus.INSUFFICIENT_EVIDENCE
    assert qual.evidence_sufficiency == FitLevel.LOW


def test_evidence_within_90_days_high_timing():
    candidate = _create_sample_candidate("FreshCo")
    recent_date = datetime.now(UTC) - timedelta(days=20)
    ev1 = _create_sample_evidence(
        claim="FreshCo hiring AI Engineer",
        published_at=recent_date,
    )
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="FreshCo",
        evidence=[ev1],
        technology_signals=["AI Engineer"],
        business_signals=["hiring"],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    agent = QualificationAgent()
    qual = agent.qualify(candidate, research, constraints=Constraints(recency_days=90))

    assert qual.timing == FitLevel.HIGH
    assert ev1.id in qual.supporting_evidence_ids


def test_evidence_older_than_90_days_timing_constraint():
    candidate = _create_sample_candidate("StaleCo")
    old_date = datetime.now(UTC) - timedelta(days=120)
    ev1 = _create_sample_evidence(
        claim="StaleCo hired developer last quarter",
        published_at=old_date,
        recency_score=40,
    )
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="StaleCo",
        evidence=[ev1],
        technology_signals=["developer"],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    agent = QualificationAgent()
    qual = agent.qualify(candidate, research, constraints=Constraints(recency_days=90))

    assert qual.timing == FitLevel.LOW


def test_empty_operator_profile_capability_unknown():
    candidate = _create_sample_candidate("Acme AI")
    ev1 = _create_sample_evidence(claim="Acme AI is adopting LangGraph workflows")
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="Acme AI",
        evidence=[ev1],
        technology_signals=["LangGraph workflows"],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    agent = QualificationAgent()
    qual = agent.qualify(candidate, research, operator_profile=OperatorProfile())

    assert qual.capability_fit == FitLevel.UNKNOWN
    assert any("no configured capabilities" in r.lower() for r in qual.reasons)


def test_populated_operator_profile_matches():
    candidate = _create_sample_candidate("Acme AI")
    ev1 = _create_sample_evidence(claim="Acme AI is adopting LangGraph workflows and AI agents")
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="Acme AI",
        evidence=[ev1],
        technology_signals=["LangGraph workflows", "AI agents"],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    profile = OperatorProfile(capabilities=["LangGraph", "AI agents", "Python"])
    agent = QualificationAgent()
    qual = agent.qualify(candidate, research, operator_profile=profile)

    assert qual.capability_fit == FitLevel.HIGH
    assert any("LangGraph" in r for r in qual.reasons)


def test_reasons_reference_valid_evidence_ids():
    candidate = _create_sample_candidate("TargetCo")
    ev1 = _create_sample_evidence(claim="TargetCo hiring staff AI engineer", recency_score=90)
    ev2 = _create_sample_evidence(claim="TargetCo official announcement regarding agentic systems", recency_score=90)
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="TargetCo",
        evidence=[ev1, ev2],
        technology_signals=["agentic systems"],
        business_signals=["hiring staff AI engineer"],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    agent = QualificationAgent()
    qual = agent.qualify(candidate, research)

    # Every ID in supporting_evidence_ids must belong to research.evidence
    actual_ids = {ev1.id, ev2.id}
    assert len(qual.supporting_evidence_ids) > 0
    for ev_id in qual.supporting_evidence_ids:
        assert ev_id in actual_ids


def test_unsupported_claims_not_qualified():
    candidate = _create_sample_candidate("VagueCo")
    ev1 = _create_sample_evidence(claim="VagueCo posted a holiday greeting")
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="VagueCo",
        evidence=[ev1],
        technology_signals=[],
        business_signals=[],
        problem_signals=[],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    agent = QualificationAgent()
    qual = agent.qualify(candidate, research)

    # Should not qualify without factual need/commercial evidence
    assert qual.qualification_status != QualificationStatus.QUALIFIED
    # Qualification text must never make unsupported definitive sales claims
    for r in qual.reasons:
        assert "definitely a client" not in r
        assert "will buy" not in r


def test_platform_integration_partner_classification():
    candidate = _create_sample_candidate("CloudPlatform Inc")
    ev1 = _create_sample_evidence(
        claim="CloudPlatform Inc announces technology partner program for agentic automation integrations",
        source_type=SourceType.OFFICIAL_ANNOUNCEMENT,
    )
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="CloudPlatform Inc",
        evidence=[ev1],
        technology_signals=["agentic automation"],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    agent = QualificationAgent()
    qual = agent.qualify(candidate, research)

    assert qual.prospect_type == ProspectType.POTENTIAL_PARTNER
    assert qual.qualification_status == QualificationStatus.WATCHLIST
    assert ev1.id in qual.supporting_evidence_ids


def test_sparse_neutral_evidence_unknown_prospect_type():
    candidate = _create_sample_candidate("NeutralCo")
    ev1 = _create_sample_evidence(claim="NeutralCo has an office in Seattle")
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="NeutralCo",
        evidence=[ev1],
        technology_signals=[],
        business_signals=[],
        problem_signals=[],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    agent = QualificationAgent()
    qual = agent.qualify(candidate, research)

    assert qual.prospect_type == ProspectType.UNKNOWN
    assert qual.qualification_status in (QualificationStatus.WATCHLIST, QualificationStatus.INSUFFICIENT_EVIDENCE)


def test_bedrock_ocean_exploration_acceptance_scenario():
    """Verify Section Acceptance Scenario A for Bedrock Ocean Exploration:

    Research includes Staff AI Platform Engineer hiring signal, official website,
    and agentic infrastructure evidence.
    Expected: QUALIFIED, POTENTIAL_CUSTOMER, evidence-backed reasons without unsupported claims.
    """
    candidate = _create_sample_candidate("Bedrock Ocean Exploration")
    ev_job = _create_sample_evidence(
        claim="Bedrock Ocean Exploration has an open role for a Staff AI Platform Engineer to build autonomous agent data processing systems.",
        source_url="https://jobs.ashbyhq.com/bedrockocean/123",
        source_type=SourceType.JOB_BOARD,
        recency_score=90,
    )
    ev_site = _create_sample_evidence(
        claim="Bedrock Ocean Exploration official website identified at https://bedrockocean.com/",
        source_url="https://bedrockocean.com",
        source_type=SourceType.COMPANY_WEBSITE,
        recency_score=100,
    )
    research = ResearchResult(
        candidate_id="cand-1",
        company_name="Bedrock Ocean Exploration",
        website=HttpUrl("https://bedrockocean.com"),
        location="Richmond, CA",
        evidence=[ev_job, ev_site],
        technology_signals=["autonomous agent data processing systems"],
        business_signals=["open role for a Staff AI Platform Engineer"],
        problem_signals=["requires scalable autonomous data processing pipelines"],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )

    agent = QualificationAgent()
    qual = agent.qualify(candidate, research)

    assert qual.qualification_status == QualificationStatus.QUALIFIED
    assert qual.prospect_type == ProspectType.POTENTIAL_CUSTOMER
    assert qual.need_fit == FitLevel.HIGH
    assert qual.commercial_relevance == FitLevel.HIGH
    assert qual.evidence_sufficiency == FitLevel.HIGH
    assert ev_job.id in qual.supporting_evidence_ids

    # Must be explainable without definitive claims
    for r in qual.reasons:
        assert "definitely a client" not in r


def test_graph_executes_qualification_node():
    input_data = AOIInput(objective=BusinessObjective(description="Find companies with AI needs."))
    state = create_initial_state(input_data)

    cand = Candidate(
        candidate_id="cand-graph-1",
        company_name="RoboNav",
        website=HttpUrl("https://robonav.ai"),
        discovery_strategy="hiring",
        discovery_reason="Found open AI role",
        source_urls=[HttpUrl("https://jobs.robonav.ai/1")],
        discovered_at=datetime.now(UTC),
    )
    state = state.model_copy(update={"candidates": [cand]})

    result = aoi_graph.invoke(state)

    assert result["status"] in (RunStatus.QUALIFYING, RunStatus.VERIFYING, RunStatus.SCORING)
    assert len(result["qualification_results"]) == 1
    assert result["qualification_results"][0].company_name == "RoboNav"
    assert result["qualification_results"][0].qualification_status in (
        QualificationStatus.QUALIFIED,
        QualificationStatus.WATCHLIST,
    )


def test_graph_runs_without_external_services_and_handles_empty():
    input_data = AOIInput(objective=BusinessObjective(description="Find companies with recent AI automation needs."))
    state = create_initial_state(input_data)
    result = aoi_graph.invoke(state)

    assert result["status"] == RunStatus.DISCOVERING
    assert result["discovery_plan"] is not None
    assert len(result["discovery_plan"].strategies) == 5
    assert len(result["qualification_results"]) == 0
