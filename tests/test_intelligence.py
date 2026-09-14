from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from aoi.agents.intelligence import IntelligenceQualityAgent
from aoi.graph.state import AOIState
from aoi.graph.workflow import aoi_graph
from aoi.report import ReportBuilder
from aoi.schemas.common import Priority, RunStatus, Source, SourceType, VerificationStatus
from aoi.schemas.discovery import Candidate
from aoi.schemas.intelligence import (
    IntelligenceQuality,
    IntelligenceQualityResult,
    IntelligenceQualityStatus,
    OpportunityNarrative,
)
from aoi.schemas.objective import AOIInput, BusinessObjective, Constraints, OperatorProfile
from aoi.schemas.qualification import (
    FitLevel,
    ProspectType,
    QualificationResult,
    QualificationStatus,
)
from aoi.schemas.report import ReportOpportunity
from aoi.schemas.research import Evidence, ResearchResult
from aoi.schemas.scoring import ScoreBreakdown, ScoringResult
from aoi.schemas.verification import VerificationClaim, VerificationResult


def make_candidate(
    candidate_id: str = "cand-1",
    company_name: str = "Acme Robotics",
    website: str | None = "https://acme-robotics.com",
    location: str = "Austin, TX",
    strategy: str = "autonomous_systems",
    reason: str = "Autonomous navigation provider",
) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        company_name=company_name,
        website=website,
        location=location,
        discovery_strategy=strategy,
        discovery_reason=reason,
        discovered_at=datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
    )


def make_evidence(
    claim: str = "Acme develops autonomous navigation for logistics",
    url: str = "https://acme-robotics.com/news/1",
    reliability: int = 90,
    recency: int = 85,
    status: VerificationStatus = VerificationStatus.VERIFIED,
    published_at: datetime | None = datetime(2026, 8, 1, 10, 0, tzinfo=UTC),
) -> Evidence:
    return Evidence(
        id=f"ev-{uuid4().hex[:8]}",
        claim=claim,
        source=Source(
            url=url,
            source_type=SourceType.COMPANY_WEBSITE,
            accessed_at=datetime(2026, 9, 14, 12, 0, tzinfo=UTC),
            published_at=published_at,
            reliability=reliability,
        ),
        evidence_summary="Article describing autonomous navigation solutions.",
        verification_status=status,
        recency_score=recency,
    )


def make_research(
    candidate_id: str = "cand-1",
    company_name: str = "Acme Robotics",
    evidence: list[Evidence] | None = None,
    problems: list[str] | None = None,
    tech: list[str] | None = None,
    business: list[str] | None = None,
    website: str | None = "https://acme-robotics.com",
) -> ResearchResult:
    evs = evidence if evidence is not None else [
        make_evidence(claim="Acme needs scalable pipeline orchestration", url="https://acme-robotics.com/blog/1"),
        make_evidence(claim="Acme uses ROS2 and PyTorch for perception", url="https://acme-robotics.com/blog/2"),
    ]
    return ResearchResult(
        candidate_id=candidate_id,
        company_name=company_name,
        website=website,
        location="Austin, TX",
        industry="Robotics",
        technology_signals=tech if tech is not None else ["ROS2", "PyTorch", "AI Perception"],
        business_signals=business if business is not None else ["Expanded engineering team", "Series A funding"],
        problem_signals=problems if problems is not None else ["Lack of robust data pipeline orchestration", "Inference latency bottlenecks"],
        evidence=evs,
        researched_at=datetime(2026, 9, 14, 12, 5, tzinfo=UTC),
    )


def make_qualification(
    candidate_id: str = "cand-1",
    company_name: str = "Acme Robotics",
    status: QualificationStatus = QualificationStatus.QUALIFIED,
    prospect_type: ProspectType = ProspectType.POTENTIAL_CUSTOMER,
    need_fit: FitLevel = FitLevel.HIGH,
    capability_fit: FitLevel = FitLevel.HIGH,
    timing: FitLevel = FitLevel.HIGH,
) -> QualificationResult:
    return QualificationResult(
        candidate_id=candidate_id,
        company_name=company_name,
        qualification_status=status,
        prospect_type=prospect_type,
        need_fit=need_fit,
        capability_fit=capability_fit,
        commercial_relevance=FitLevel.HIGH,
        timing=timing,
        evidence_sufficiency=FitLevel.HIGH,
        reasons=["Clear technical need in data pipelines", "Active logistics deployment"],
        supporting_evidence_ids=["ev-1", "ev-2"],
        qualified_at=datetime(2026, 9, 14, 12, 10, tzinfo=UTC),
    )


def make_verification(
    candidate_id: str = "cand-1",
    company_name: str = "Acme Robotics",
    status: VerificationStatus = VerificationStatus.VERIFIED,
    contradicted: list[VerificationClaim] | None = None,
) -> VerificationResult:
    v_claims = [
        VerificationClaim(
            claim="Acme needs scalable pipeline orchestration",
            status=VerificationStatus.VERIFIED,
            supporting_evidence_ids=["ev-1"],
            explanation="Corroborated by official job post and blog.",
        ),
        VerificationClaim(
            claim="Acme uses ROS2 and PyTorch for perception",
            status=VerificationStatus.VERIFIED,
            supporting_evidence_ids=["ev-2"],
            explanation="Corroborated by technical architecture whitepaper.",
        ),
    ]
    return VerificationResult(
        candidate_id=candidate_id,
        company_name=company_name,
        verification_status=status,
        verified_claims=v_claims if status == VerificationStatus.VERIFIED else [],
        contradicted_claims=contradicted or [],
        evidence_checked=2,
        independent_sources=2,
        verified_at=datetime(2026, 9, 14, 12, 15, tzinfo=UTC),
    )


def make_scoring(
    candidate_id: str = "cand-1",
    company_name: str = "Acme Robotics",
    priority: Priority = Priority.HIGH,
    opportunity_score: float = 85.0,
    confidence_score: float = 88.0,
    verification_status: VerificationStatus = VerificationStatus.VERIFIED,
) -> ScoringResult:
    breakdown = ScoreBreakdown(
        need_fit=88.0,
        capability_fit=82.0,
        evidence_strength=90.0,
        timing=80.0,
        commercial_potential=85.0,
        accessibility=85.0,
        opportunity_score=opportunity_score,
        confidence_score=confidence_score,
        priority=priority,
    )
    return ScoringResult(
        candidate_id=candidate_id,
        company_name=company_name,
        scores=breakdown,
        scoring_reasons=["Verified data orchestration need", "High capability fit with operator"],
        scoring_warnings=[],
        verification_status=verification_status,
        scored_at=datetime(2026, 9, 14, 12, 20, tzinfo=UTC),
    )


def make_operator_profile() -> OperatorProfile:
    return OperatorProfile(
        capabilities=["AI Data Pipeline Engineering", "Latency Optimization", "MLOps"],
        portfolio_projects=["RoboPipe Architecture"],
        preferred_services=["Pipeline Audit & Build"],
    )


# ==============================================================================
# 1. Schemas & Status Thresholds
# ==============================================================================

def test_intelligence_quality_status_thresholds():
    assert IntelligenceQualityStatus.from_score(100.0) == IntelligenceQualityStatus.HIGH
    assert IntelligenceQualityStatus.from_score(80.0) == IntelligenceQualityStatus.HIGH
    assert IntelligenceQualityStatus.from_score(79.9) == IntelligenceQualityStatus.GOOD
    assert IntelligenceQualityStatus.from_score(65.0) == IntelligenceQualityStatus.GOOD
    assert IntelligenceQualityStatus.from_score(64.9) == IntelligenceQualityStatus.LIMITED
    assert IntelligenceQualityStatus.from_score(50.0) == IntelligenceQualityStatus.LIMITED
    assert IntelligenceQualityStatus.from_score(49.9) == IntelligenceQualityStatus.INSUFFICIENT
    assert IntelligenceQualityStatus.from_score(0.0) == IntelligenceQualityStatus.INSUFFICIENT


def test_schema_validation_valid():
    q = IntelligenceQuality(
        completeness_score=85.0,
        evidence_coverage_score=80.0,
        consistency_score=90.0,
        actionability_score=75.0,
        quality_score=82.5,
        warnings=[],
    )
    narrative = OpportunityNarrative(
        why_company="Surfaced via search.",
        why_now="Recent news.",
        identified_problem="High latency.",
        why_fit="Matches MLOps capabilities.",
        evidence_summary="Verified claims.",
        next_human_step="Human outreach review.",
    )
    res = IntelligenceQualityResult(
        candidate_id="cand-1",
        company_name="Acme",
        quality=q,
        narrative=narrative,
        strengths=["Strong evidence"],
        weaknesses=[],
        converging_signals=["HIRING and TECH"],
        material_claims_covered=5,
        material_claims_total=5,
        quality_warnings=[],
        generated_at=datetime.now(UTC),
    )
    assert res.candidate_id == "cand-1"
    assert res.quality.quality_score == 82.5


def test_schema_bounds_enforced():
    with pytest.raises(ValidationError):
        IntelligenceQuality(
            completeness_score=105.0,  # exceeds 100
            evidence_coverage_score=80.0,
            consistency_score=90.0,
            actionability_score=75.0,
            quality_score=82.5,
        )

    with pytest.raises(ValidationError):
        IntelligenceQuality(
            completeness_score=-5.0,  # below 0
            evidence_coverage_score=80.0,
            consistency_score=90.0,
            actionability_score=75.0,
            quality_score=82.5,
        )


# ==============================================================================
# 2. Completeness Calculation
# ==============================================================================

def test_completeness_all_fields_present():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    res.people = [{"name": "Jane Doe", "role": "VP Engineering", "profile_url": None}]
    qual = make_qualification()
    ver = make_verification()
    scoring = make_scoring()
    op = make_operator_profile()

    result = agent.evaluate(cand, res, qual, ver, scoring, operator_profile=op)
    assert result.quality.completeness_score == 100.0


def test_completeness_partial_fields():
    agent = IntelligenceQualityAgent()
    cand = make_candidate(location="", strategy="search")  # partial identity
    res = make_research(problems=["Need help"])  # single problem
    res.evidence = [make_evidence(published_at=None)]  # single evidence, missing date
    qual = make_qualification(need_fit=FitLevel.MEDIUM)
    scoring = make_scoring()
    scoring.scoring_reasons = ["Single reason"]

    result = agent.evaluate(cand, res, qual, None, scoring, operator_profile=None)
    # Missing verification result and operator profile reduces score deterministically
    assert result.quality.completeness_score < 80.0
    assert result.quality.completeness_score > 30.0


def test_completeness_missing_upstream_fields():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    # Missing research, qualification, verification, scoring
    result = agent.evaluate(cand, None, None, None, None)
    assert result.quality.completeness_score < 20.0
    assert "Low intelligence completeness; material candidate attributes missing." in result.quality_warnings


# ==============================================================================
# 3. Evidence Coverage Calculation
# ==============================================================================

def test_evidence_coverage_all_categories_verified():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    ver = VerificationResult(
        candidate_id="cand-1",
        company_name="Acme",
        verification_status=VerificationStatus.VERIFIED,
        verified_claims=[
            VerificationClaim(claim="Need: data pipeline problem", status=VerificationStatus.VERIFIED, explanation="corroborated"),
            VerificationClaim(claim="Tech: uses AI tools", status=VerificationStatus.VERIFIED, explanation="corroborated"),
            VerificationClaim(claim="Timing: recent expansion", status=VerificationStatus.VERIFIED, explanation="corroborated"),
            VerificationClaim(claim="Commercial: B2B client base", status=VerificationStatus.VERIFIED, explanation="corroborated"),
        ],
        verified_at=datetime.now(UTC),
    )
    qual = make_qualification()
    res = make_research()
    scoring = make_scoring()

    result = agent.evaluate(cand, res, qual, ver, scoring)
    assert result.quality.evidence_coverage_score == 100.0
    assert result.material_claims_covered == 5
    assert result.material_claims_total == 5


def test_evidence_coverage_deduplication():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    # Duplicate URLs from same domain
    ev1 = make_evidence(url="https://acme.com/news/1")
    ev2 = make_evidence(url="https://acme.com/news/1/")  # trailing slash variant
    res = make_research(evidence=[ev1, ev2])
    ver = make_verification()
    qual = make_qualification()
    scoring = make_scoring()

    result = agent.evaluate(cand, res, qual, ver, scoring)
    # Deduplication ensures duplicate sources do not inflate strengths
    assert not any("Supported by 2 independent unique evidence sources" in s for s in result.strengths)


def test_evidence_coverage_contradiction_detection():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    contradicted_claim = VerificationClaim(
        claim="Acme needs cloud data pipelines",
        status=VerificationStatus.CONTRADICTED,
        explanation="Official company filing states all systems are fully on-prem and locked.",
    )
    ver = make_verification(
        status=VerificationStatus.CONTRADICTED,
        contradicted=[contradicted_claim],
    )
    qual = make_qualification()
    res = make_research()
    scoring = make_scoring(priority=Priority.DISCARD, verification_status=VerificationStatus.CONTRADICTED)

    result = agent.evaluate(cand, res, qual, ver, scoring)
    expected_msg = "Material evidence contradiction detected for Acme needs cloud data pipelines."
    assert expected_msg in result.quality_warnings
    assert expected_msg in result.weaknesses
    # Contradiction sets need coverage to 0
    assert result.quality.evidence_coverage_score < 75.0


def test_evidence_coverage_missing_timing():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research(evidence=[make_evidence(published_at=None)])
    qual = make_qualification(timing=FitLevel.LOW)
    ver = make_verification()
    scoring = make_scoring()

    result = agent.evaluate(cand, res, qual, ver, scoring)
    assert "No timing or urgency evidence available." in result.weaknesses


# ==============================================================================
# 4. Consistency Calculation
# ==============================================================================

def test_consistency_all_consistent():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    qual = make_qualification()
    ver = make_verification()
    scoring = make_scoring()

    result = agent.evaluate(cand, res, qual, ver, scoring)
    assert result.quality.consistency_score == 100.0


def test_consistency_prospect_type_contradiction():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    qual = make_qualification(prospect_type=ProspectType.COMPETITOR_OR_VENDOR)
    ver = make_verification()
    scoring = make_scoring(priority=Priority.HIGH)  # Contradiction: competitor scored HIGH priority

    result = agent.evaluate(cand, res, qual, ver, scoring)
    assert result.quality.consistency_score < 100.0
    assert any("Prospect-type contradiction" in w for w in result.quality_warnings)


def test_consistency_prospect_type_unknown():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    qual = make_qualification(prospect_type=ProspectType.UNKNOWN)
    ver = make_verification()
    scoring = make_scoring()

    result = agent.evaluate(cand, res, qual, ver, scoring)
    assert result.quality.consistency_score == 87.5  # 50% on 25% weight = 12.5 loss
    assert any("Prospect-type consistency gap" in w for w in result.weaknesses)


def test_consistency_need_contradiction():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    qual = make_qualification()
    contradiction = VerificationClaim(
        claim="Acme need for data orchestration",
        status=VerificationStatus.CONTRADICTED,
        explanation="Company refutes this need.",
    )
    ver = make_verification(status=VerificationStatus.CONTRADICTED, contradicted=[contradiction])
    scoring = make_scoring(verification_status=VerificationStatus.CONTRADICTED, priority=Priority.DISCARD)

    result = agent.evaluate(cand, res, qual, ver, scoring)
    assert any("Need consistency contradiction" in w for w in result.quality_warnings)


def test_consistency_verification_scoring_contradiction():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    qual = make_qualification()
    ver = make_verification(status=VerificationStatus.CONTRADICTED)
    scoring = make_scoring(priority=Priority.HIGH, verification_status=VerificationStatus.CONTRADICTED)

    result = agent.evaluate(cand, res, qual, ver, scoring)
    assert any("Verification/scoring contradiction" in w for w in result.quality_warnings)


# ==============================================================================
# 5. Actionability Calculation
# ==============================================================================

def test_actionability_strong_decision():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    res.people = [{"name": "Jane Doe", "role": "CTO", "profile_url": None}]
    qual = make_qualification()
    ver = make_verification()
    scoring = make_scoring()
    op = make_operator_profile()

    result = agent.evaluate(cand, res, qual, ver, scoring, operator_profile=op)
    assert result.quality.actionability_score == 100.0


def test_actionability_weights_and_rounding():
    agent = IntelligenceQualityAgent()
    cand = make_candidate(website=None)
    res = make_research(website=None, problems=["One problem"])
    qual = make_qualification()
    ver = make_verification()
    scoring = make_scoring()

    result = agent.evaluate(cand, res, qual, ver, scoring)
    # Missing website and single problem produces intermediate score with 1 decimal precision
    assert result.quality.actionability_score == round(result.quality.actionability_score, 1)
    assert result.quality.actionability_score < 100.0


def test_actionability_company_accessibility():
    agent = IntelligenceQualityAgent()
    # Missing website and people
    cand = make_candidate(website=None)
    res = make_research(website=None)
    res.people = []
    qual = make_qualification()
    ver = make_verification()
    scoring = make_scoring()

    result = agent.evaluate(cand, res, qual, ver, scoring)
    assert "Limited company accessibility: missing verified website." in result.weaknesses


# ==============================================================================
# 6. Overall Quality Score & Status
# ==============================================================================

def test_quality_score_exact_formula():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    res.people = [{"name": "Jane Doe", "role": "CTO", "profile_url": None}]
    qual = make_qualification()
    ver = make_verification()
    scoring = make_scoring()
    op = make_operator_profile()

    result = agent.evaluate(cand, res, qual, ver, scoring, operator_profile=op)
    q = result.quality
    expected_quality = round(
        q.completeness_score * 0.25
        + q.evidence_coverage_score * 0.30
        + q.consistency_score * 0.20
        + q.actionability_score * 0.25,
        1,
    )
    assert q.quality_score == expected_quality
    assert IntelligenceQualityStatus.from_score(q.quality_score) == IntelligenceQualityStatus.HIGH


# ==============================================================================
# 7. Signal Synthesis
# ==============================================================================

def test_signal_synthesis_two_distinct_categories():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research(
        tech=["PyTorch AI model"],
        business=["Hiring senior AI engineers"],
        problems=[],
    )
    ver = make_verification()
    qual = make_qualification()
    scoring = make_scoring()

    result = agent.evaluate(cand, res, qual, ver, scoring)
    assert len(result.converging_signals) == 1
    assert "HIRING" in result.converging_signals[0]
    assert "TECHNOLOGY" in result.converging_signals[0]


def test_signal_synthesis_single_category_no_synthesis():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    # Only single technology signal without others
    res = make_research(
        evidence=[make_evidence(claim="Acme uses Python")],
        tech=["Python"],
        business=[],
        problems=[],
    )
    ver = make_verification()
    qual = make_qualification()
    scoring = make_scoring()

    result = agent.evaluate(cand, res, qual, ver, scoring)
    assert result.converging_signals == []


def test_signal_synthesis_contradiction_suppression():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research(tech=["AI"], business=["Hiring engineers"])
    contradiction = VerificationClaim(
        claim="Acme AI claims",
        status=VerificationStatus.CONTRADICTED,
        explanation="Contradicted",
    )
    ver = make_verification(status=VerificationStatus.CONTRADICTED, contradicted=[contradiction])
    qual = make_qualification()
    scoring = make_scoring(priority=Priority.DISCARD, verification_status=VerificationStatus.CONTRADICTED)

    result = agent.evaluate(cand, res, qual, ver, scoring)
    assert result.converging_signals == []


# ==============================================================================
# 8. Opportunity Narratives
# ==============================================================================

def test_narrative_why_company():
    agent = IntelligenceQualityAgent()
    cand = make_candidate(strategy="b2b_logistics", reason="Autonomous drone provider")
    res = make_research()
    result = agent.evaluate(cand, res, None, None, None)
    assert "surfaced via b2b_logistics" in result.narrative.why_company
    assert "Autonomous drone provider" in result.narrative.why_company


def test_narrative_why_now_present():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    ev = make_evidence(published_at=datetime(2026, 9, 1, tzinfo=UTC), claim="Acme launched pilot")
    res = make_research(evidence=[ev])
    result = agent.evaluate(cand, res, None, None, None)
    assert "Active timing signals observed" in result.narrative.why_now
    assert "2026-09-01" in result.narrative.why_now


def test_narrative_why_now_fallback():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research(evidence=[make_evidence(published_at=None)])
    qual = make_qualification(timing=FitLevel.LOW)
    result = agent.evaluate(cand, res, qual, None, None)
    assert result.narrative.why_now == "No recent timing signal identified."


def test_narrative_identified_problem_present():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research(problems=["High query latency", "Memory leaks"])
    result = agent.evaluate(cand, res, None, None, None)
    assert result.narrative.identified_problem == "High query latency; Memory leaks"


def test_narrative_identified_problem_none():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research(problems=[])
    qual = make_qualification()
    qual.reasons = []
    result = agent.evaluate(cand, res, qual, None, None)
    assert result.narrative.identified_problem is None


def test_narrative_why_fit_with_operator_profile():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research(problems=["Database latency"])
    op = make_operator_profile()
    result = agent.evaluate(cand, res, None, None, None, operator_profile=op)
    assert "AI Data Pipeline Engineering" in result.narrative.why_fit
    assert "Database latency" in result.narrative.why_fit


def test_narrative_why_fit_empty_operator_profile():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    result = agent.evaluate(cand, res, None, None, None, operator_profile=None)
    assert (
        result.narrative.why_fit
        == "Operator profile not configured; fit cannot be determined without defined capabilities."
    )


def test_narrative_next_human_step_high_priority():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    ver = make_verification()
    scoring = make_scoring(priority=Priority.HIGH)
    result = agent.evaluate(cand, res, None, ver, scoring)
    assert "Operator should conduct manual outreach preparation" in result.narrative.next_human_step


def test_narrative_next_human_step_contradicted():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    ver = make_verification(status=VerificationStatus.CONTRADICTED)
    scoring = make_scoring(priority=Priority.DISCARD, verification_status=VerificationStatus.CONTRADICTED)
    result = agent.evaluate(cand, res, None, ver, scoring)
    assert result.narrative.next_human_step == "Resolve the conflicting evidence before considering this opportunity."


def test_narrative_next_human_step_disqualified():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    qual = make_qualification(status=QualificationStatus.DISQUALIFIED)
    scoring = make_scoring(priority=Priority.DISCARD)
    result = agent.evaluate(cand, res, qual, None, scoring)
    assert "candidate was disqualified during qualification" in result.narrative.next_human_step


# ==============================================================================
# 9. Determinism Verification
# ==============================================================================

def test_determinism_identical_output():
    agent = IntelligenceQualityAgent()
    cand = make_candidate()
    res = make_research()
    qual = make_qualification()
    ver = make_verification()
    scoring = make_scoring()
    op = make_operator_profile()

    run1 = agent.evaluate(cand, res, qual, ver, scoring, operator_profile=op)
    run2 = agent.evaluate(cand, res, qual, ver, scoring, operator_profile=op)

    assert run1.model_dump(exclude={"generated_at"}) == run2.model_dump(exclude={"generated_at"})


# ==============================================================================
# 10. Graph and Report Integration
# ==============================================================================

def test_graph_populates_quality_results_and_completes():
    cand = make_candidate()
    state = AOIState(
        run_id="AOI-TEST-GRAPH",
        input=AOIInput(
            objective=BusinessObjective(description="Find B2B robotics adopting AI navigation"),
            operator_profile=make_operator_profile(),
            constraints=Constraints(),
        ),
        candidates=[cand],
    )

    final_state = aoi_graph.invoke(state)
    assert final_state["status"] == RunStatus.COMPLETED
    assert len(final_state["quality_results"]) == 1
    qr = final_state["quality_results"][0]
    assert qr.candidate_id == cand.candidate_id
    assert qr.company_name == cand.company_name
    assert qr.quality.quality_score > 0.0


def test_graph_empty_candidates():
    state = AOIState(
        run_id="AOI-EMPTY",
        input=AOIInput(
            objective=BusinessObjective(description="Find empty candidates test"),
            constraints=Constraints(),
        ),
        candidates=[],
    )
    final_state = aoi_graph.invoke(state)
    assert final_state["status"] == RunStatus.COMPLETED
    assert final_state["quality_results"] == []
    assert final_state["report"].opportunities == []


def test_graph_partial_state():
    from aoi.graph.workflow import reporting_node

    cand = make_candidate()
    res = make_research()
    scoring = make_scoring()
    agent = IntelligenceQualityAgent()
    qr = agent.evaluate(cand, res, None, None, scoring)
    state = AOIState(
        run_id="AOI-PARTIAL",
        input=AOIInput(
            objective=BusinessObjective(description="Find partial pipeline test"),
            constraints=Constraints(),
        ),
        status=RunStatus.PARTIAL,
        warnings=["Upstream failure simulated"],
        candidates=[cand],
        scoring_results=[scoring],
        quality_results=[qr],
    )
    result = reporting_node(state)
    assert result["status"] == RunStatus.PARTIAL
    assert result["report"].run_status == RunStatus.PARTIAL


def test_report_builder_opportunity_depth_integration():
    builder = ReportBuilder()
    cand = make_candidate()
    res = make_research()
    qual = make_qualification()
    ver = make_verification()
    scoring = make_scoring()
    agent = IntelligenceQualityAgent()
    qr = agent.evaluate(cand, res, qual, ver, scoring, operator_profile=make_operator_profile())

    report = builder.build(
        run_id="AOI-REPORT-1",
        objective={"description": "Test objective"},
        candidates=[cand],
        research_results=[res],
        qualification_results=[qual],
        verification_results=[ver],
        scoring_results=[scoring],
        quality_results=[qr],
    )

    assert len(report.opportunities) == 1
    opp = report.opportunities[0]
    assert isinstance(opp, ReportOpportunity)
    assert opp.quality is not None
    assert opp.quality.quality_score == qr.quality.quality_score
    assert opp.narrative is not None
    assert opp.narrative.why_company == qr.narrative.why_company
    assert opp.quality_status == IntelligenceQualityStatus.HIGH
    assert len(opp.strengths) > 0


def test_upstream_scores_and_statuses_strictly_unchanged():
    builder = ReportBuilder()
    cand = make_candidate()
    res = make_research()
    qual = make_qualification()
    ver = make_verification()
    scoring = make_scoring(opportunity_score=83.4, confidence_score=78.2, priority=Priority.QUALIFIED)

    agent = IntelligenceQualityAgent()
    qr = agent.evaluate(cand, res, qual, ver, scoring)

    report = builder.build(
        candidates=[cand],
        research_results=[res],
        qualification_results=[qual],
        verification_results=[ver],
        scoring_results=[scoring],
        quality_results=[qr],
    )

    opp = report.opportunities[0]
    # Upstream values must be strictly preserved
    assert opp.opportunity_score == 83.4
    assert opp.confidence_score == 78.2
    assert opp.priority == Priority.QUALIFIED
    assert opp.qualification_status == QualificationStatus.QUALIFIED
    assert opp.verification_status == VerificationStatus.VERIFIED
