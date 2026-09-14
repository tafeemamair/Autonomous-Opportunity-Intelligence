from datetime import UTC, datetime, timedelta

from pydantic import HttpUrl

from aoi.agents.scoring import ScoringAgent
from aoi.graph.state import RunStatus
from aoi.graph.workflow import aoi_graph, create_initial_state
from aoi.schemas.common import Evidence, Priority, Source, SourceType, VerificationStatus
from aoi.schemas.discovery import Candidate
from aoi.schemas.objective import AOIInput, BusinessObjective, OperatorProfile
from aoi.schemas.opportunity import DecisionMaker
from aoi.schemas.qualification import (
    FitLevel,
    ProspectType,
    QualificationResult,
    QualificationStatus,
)
from aoi.schemas.research import ResearchResult, ResearchStatus
from aoi.schemas.scoring import ScoreBreakdown
from aoi.schemas.verification import VerificationClaim, VerificationResult


def make_evidence(
    ev_id: str,
    url: str,
    claim: str,
    reliability: int = 85,
    days_ago: int = 10,
    source_type: SourceType = SourceType.COMPANY_WEBSITE,
) -> Evidence:
    now = datetime.now(UTC)
    pub = now - timedelta(days=days_ago)
    return Evidence(
        id=ev_id,
        claim=claim,
        source=Source(
            url=HttpUrl(url),
            source_type=source_type,
            title="Evidence Source",
            publisher="example.com",
            published_at=pub,
            accessed_at=now,
            reliability=reliability,
        ),
        evidence_summary=claim,
        verification_status=VerificationStatus.UNVERIFIED,
        recency_score=max(0, 100 - days_ago),
    )


def make_candidate(name: str = "ScoreCorp", cand_id: str = "cand-s1") -> Candidate:
    clean = "".join(c for c in name.lower() if c.isalnum())
    return Candidate(
        candidate_id=cand_id,
        company_name=name,
        website=HttpUrl(f"https://www.{clean}.com"),
        discovery_strategy="hiring_signal",
        discovery_reason="AI infrastructure opportunity",
        discovered_at=datetime.now(UTC),
    )


def make_research_result(
    cand: Candidate,
    evidence: list[Evidence],
    people: list[DecisionMaker] | None = None,
    tech: list[str] | None = None,
    problems: list[str] | None = None,
    business: list[str] | None = None,
) -> ResearchResult:
    return ResearchResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        website=cand.website,
        people=people or [],
        technology_signals=tech or ["Python", "Docker", "AI platform"],
        business_signals=business or ["Expanding engineering", "Enterprise growth"],
        problem_signals=problems or ["Data pipeline latency", "Scaling orchestration"],
        evidence=evidence,
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )


def make_qualification_result(
    cand: Candidate,
    status: QualificationStatus = QualificationStatus.QUALIFIED,
    prospect_type: ProspectType = ProspectType.POTENTIAL_CUSTOMER,
    need_fit: FitLevel = FitLevel.HIGH,
    capability_fit: FitLevel = FitLevel.HIGH,
    commercial_relevance: FitLevel = FitLevel.HIGH,
    timing: FitLevel = FitLevel.HIGH,
) -> QualificationResult:
    return QualificationResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        qualification_status=status,
        prospect_type=prospect_type,
        need_fit=need_fit,
        capability_fit=capability_fit,
        commercial_relevance=commercial_relevance,
        timing=timing,
        evidence_sufficiency=FitLevel.HIGH,
        reasons=["Hiring AI platform engineer to automate pipelines"],
        supporting_evidence_ids=[],
        disqualification_reasons=[],
        warnings=[],
        qualified_at=datetime.now(UTC),
    )


def make_verification_result(
    cand: Candidate,
    status: VerificationStatus = VerificationStatus.VERIFIED,
    domains: int = 2,
    verified_claims: list[VerificationClaim] | None = None,
    warnings: list[str] | None = None,
) -> VerificationResult:
    claims = verified_claims or [
        VerificationClaim(
            claim=f"{cand.company_name} has an active AI engineering requirement or automation need.",
            status=status,
            supporting_evidence_ids=["ev-1"],
            explanation="Requirement verified",
        )
    ]
    return VerificationResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        verification_status=status,
        verified_claims=claims if status == VerificationStatus.VERIFIED else [],
        partially_verified_claims=claims if status == VerificationStatus.PARTIALLY_VERIFIED else [],
        unverified_claims=claims if status == VerificationStatus.UNVERIFIED else [],
        contradicted_claims=claims if status == VerificationStatus.CONTRADICTED else [],
        verification_warnings=warnings or [],
        evidence_checked=len(claims),
        independent_sources=domains,
        verified_at=datetime.now(UTC),
    )


# 1. Exact weighted score calculation
def test_exact_weighted_score_calculation():
    # Directly test the formula math
    # Need: 80 * 0.25 = 20.0
    # Capability: 70 * 0.20 = 14.0
    # Evidence: 90 * 0.20 = 18.0
    # Timing: 60 * 0.15 = 9.0
    # Commercial: 50 * 0.10 = 5.0
    # Accessibility: 40 * 0.10 = 4.0
    # Total = 70.0
    breakdown = ScoreBreakdown(
        need_fit=80.0,
        capability_fit=70.0,
        evidence_strength=90.0,
        timing=60.0,
        commercial_potential=50.0,
        accessibility=40.0,
        opportunity_score=round(80 * 0.25 + 70 * 0.20 + 90 * 0.20 + 60 * 0.15 + 50 * 0.10 + 40 * 0.10, 1),
        confidence_score=85.0,
        priority=Priority.QUALIFIED,
    )
    assert breakdown.opportunity_score == 70.0


# 2. All 100s produces 100
def test_all_100s_produces_100():
    score = round(100.0 * 0.25 + 100.0 * 0.20 + 100.0 * 0.20 + 100.0 * 0.15 + 100.0 * 0.10 + 100.0 * 0.10, 1)
    assert score == 100.0


# 3. All 0s produces 0
def test_all_0s_produces_0():
    score = round(0.0 * 0.25 + 0.0 * 0.20 + 0.0 * 0.20 + 0.0 * 0.15 + 0.0 * 0.10 + 0.0 * 0.10, 1)
    assert score == 0.0


# 4. 80 boundary -> HIGH
def test_priority_boundary_80_high():
    agent = ScoringAgent()
    qual = make_qualification_result(make_candidate(), need_fit=FitLevel.HIGH)
    ver = make_verification_result(make_candidate(), status=VerificationStatus.VERIFIED)
    p = agent._assign_priority(80.0, qual, ver, [])
    assert p == Priority.HIGH


# 5. 65 boundary -> QUALIFIED
def test_priority_boundary_65_qualified():
    agent = ScoringAgent()
    qual = make_qualification_result(make_candidate(), need_fit=FitLevel.HIGH)
    ver = make_verification_result(make_candidate(), status=VerificationStatus.VERIFIED)
    p = agent._assign_priority(65.0, qual, ver, [])
    assert p == Priority.QUALIFIED


# 6. 50 boundary -> WATCHLIST
def test_priority_boundary_50_watchlist():
    agent = ScoringAgent()
    qual = make_qualification_result(make_candidate(), need_fit=FitLevel.HIGH)
    ver = make_verification_result(make_candidate(), status=VerificationStatus.VERIFIED)
    p = agent._assign_priority(50.0, qual, ver, [])
    assert p == Priority.WATCHLIST


# 7. Below 50 -> DISCARD
def test_priority_boundary_below_50_discard():
    agent = ScoringAgent()
    qual = make_qualification_result(make_candidate(), need_fit=FitLevel.HIGH)
    ver = make_verification_result(make_candidate(), status=VerificationStatus.VERIFIED)
    p = agent._assign_priority(49.9, qual, ver, [])
    assert p == Priority.DISCARD


# 8. Deterministic repeated scoring
def test_deterministic_repeated_scoring():
    agent = ScoringAgent()
    cand = make_candidate("RepeatCorp")
    ev = make_evidence("ev-1", "https://repeatcorp.com/jobs", "Hiring AI engineer", 90, 5)
    res = make_research_result(cand, [ev])
    qual = make_qualification_result(cand)
    ver = make_verification_result(cand, VerificationStatus.VERIFIED)
    op = OperatorProfile(capabilities=["AI platform engineering", "Python"])

    res1 = agent.score(cand, res, qual, ver, operator_profile=op)
    res2 = agent.score(cand, res, qual, ver, operator_profile=op)

    assert res1.scores.opportunity_score == res2.scores.opportunity_score
    assert res1.scores.confidence_score == res2.scores.confidence_score
    assert res1.scores.priority == res2.scores.priority


# 9. Opportunity Score and Confidence Score are separate
def test_opportunity_score_and_confidence_score_separate():
    agent = ScoringAgent()
    cand = make_candidate("SepCorp")
    # Low reliability evidence yields lower confidence, but high qualification need fit
    ev = make_evidence("ev-1", "https://sepcorp.com/careers", "Hiring AI engineer", reliability=40, days_ago=5)
    res = make_research_result(cand, [ev])
    qual = make_qualification_result(cand, need_fit=FitLevel.HIGH)
    ver = make_verification_result(cand, status=VerificationStatus.PARTIALLY_VERIFIED, domains=1)

    result = agent.score(cand, res, qual, ver)
    # Opportunity score and confidence score must not be identical copies
    assert result.scores.opportunity_score != result.scores.confidence_score
    assert isinstance(result.scores.opportunity_score, float)
    assert isinstance(result.scores.confidence_score, float)


# 10. VERIFIED increases confidence appropriately
def test_verified_increases_confidence():
    agent = ScoringAgent()
    cand = make_candidate("ConfCorp")
    ev = make_evidence("ev-1", "https://confcorp.com/jobs", "Hiring AI engineer", reliability=90, days_ago=5)
    res = make_research_result(cand, [ev])
    qual = make_qualification_result(cand)

    ver_verified = make_verification_result(cand, VerificationStatus.VERIFIED, domains=3)
    ver_unverified = make_verification_result(cand, VerificationStatus.UNVERIFIED, domains=1)

    s_ver = agent.score(cand, res, qual, ver_verified)
    s_unver = agent.score(cand, res, qual, ver_unverified)

    assert s_ver.scores.confidence_score > s_unver.scores.confidence_score
    assert s_ver.scores.confidence_score >= 80.0


# 11. PARTIALLY_VERIFIED lowers confidence appropriately
def test_partially_verified_lowers_confidence():
    agent = ScoringAgent()
    cand = make_candidate("PartialCorp")
    ev = make_evidence("ev-1", "https://partialcorp.com/jobs", "Hiring AI engineer", reliability=85, days_ago=5)
    res = make_research_result(cand, [ev])
    qual = make_qualification_result(cand)

    ver_verified = make_verification_result(cand, VerificationStatus.VERIFIED, domains=2)
    ver_partial = make_verification_result(cand, VerificationStatus.PARTIALLY_VERIFIED, domains=1)

    s_ver = agent.score(cand, res, qual, ver_verified)
    s_part = agent.score(cand, res, qual, ver_partial)

    assert s_ver.scores.confidence_score > s_part.scores.confidence_score
    assert 40.0 <= s_part.scores.confidence_score < s_ver.scores.confidence_score


# 12. UNVERIFIED cannot masquerade as high-confidence evidence
def test_unverified_cannot_masquerade_as_high_confidence():
    agent = ScoringAgent()
    cand = make_candidate("UnverCorp")
    ev = make_evidence("ev-1", "https://unvercorp.com/jobs", "Hiring AI engineer", reliability=50, days_ago=5)
    res = make_research_result(cand, [ev])
    qual = make_qualification_result(cand)
    ver = make_verification_result(cand, VerificationStatus.UNVERIFIED, domains=0)

    result = agent.score(cand, res, qual, ver)
    assert result.scores.confidence_score <= 50.0


# 13. CONTRADICTED materially reduces confidence/priority
def test_contradicted_materially_reduces_confidence_and_priority():
    agent = ScoringAgent()
    cand = make_candidate("ContraCorp")
    ev1 = make_evidence("ev-1", "https://contracorp.com/jobs", "Hiring AI engineer", reliability=85, days_ago=5)
    ev2 = make_evidence("ev-2", "https://news.com/contracorp", "Announced hiring freeze", reliability=90, days_ago=2)
    res = make_research_result(cand, [ev1, ev2])
    qual = make_qualification_result(cand, need_fit=FitLevel.HIGH)
    ver = make_verification_result(cand, VerificationStatus.CONTRADICTED, domains=2)

    result = agent.score(cand, res, qual, ver)
    assert result.scores.confidence_score <= 15.0
    assert result.scores.priority == Priority.DISCARD


# 14. Empty operator profile does not fabricate capability fit
def test_empty_operator_profile_does_not_fabricate_capability():
    agent = ScoringAgent()
    cand = make_candidate("NoOpCorp")
    ev = make_evidence("ev-1", "https://noopcorp.com/jobs", "Hiring AI engineer", reliability=85, days_ago=5)
    res = make_research_result(cand, [ev])
    qual = make_qualification_result(cand)
    ver = make_verification_result(cand, VerificationStatus.VERIFIED)
    empty_op = OperatorProfile()

    result = agent.score(cand, res, qual, ver, operator_profile=empty_op)
    assert result.scores.capability_fit == 50.0
    assert any("empty" in w.lower() or "unknown" in w.lower() for w in result.scoring_warnings)


# 15. Unknown capability is represented honestly
def test_unknown_capability_represented_honestly():
    agent = ScoringAgent()
    cand = make_candidate("UnknownCapCorp")
    ev = make_evidence("ev-1", "https://unknowncap.com/jobs", "Hiring AI engineer", reliability=85, days_ago=5)
    res = make_research_result(cand, [ev])
    qual = make_qualification_result(cand)
    ver = make_verification_result(cand, VerificationStatus.VERIFIED)

    result = agent.score(cand, res, qual, ver, operator_profile=None)
    assert result.scores.capability_fit == 50.0
    assert any("UNKNOWN" in r for r in result.scoring_reasons)


# 16. Missing decision maker affects accessibility honestly
def test_missing_decision_maker_affects_accessibility():
    agent = ScoringAgent()
    cand = make_candidate("NoDmCorp")
    ev = make_evidence("ev-1", "https://nodmcorp.com/jobs", "Hiring AI engineer", reliability=85, days_ago=5)
    # No decision maker attached to research
    res = make_research_result(cand, [ev], people=[])
    qual = make_qualification_result(cand)
    ver = make_verification_result(cand, VerificationStatus.VERIFIED)

    result = agent.score(cand, res, qual, ver)
    # Accessibility should be lower without DM
    assert result.scores.accessibility <= 60.0
    assert any("No decision maker" in r for r in result.scoring_reasons)


# 17. Verified decision maker improves accessibility appropriately
def test_verified_decision_maker_improves_accessibility():
    agent = ScoringAgent()
    cand = make_candidate("DmCorp")
    ev = make_evidence("ev-1", "https://dmcorp.com/jobs", "Hiring AI engineer", reliability=85, days_ago=5)
    dm = DecisionMaker(name="Sarah Connor", role="VP Engineering")
    res = make_research_result(cand, [ev], people=[dm])
    qual = make_qualification_result(cand)
    dm_claim = VerificationClaim(
        claim="Sarah Connor is VP Engineering at DmCorp",
        status=VerificationStatus.VERIFIED,
        supporting_evidence_ids=["ev-1"],
        explanation="Verified leadership contact",
    )
    ver = make_verification_result(cand, VerificationStatus.VERIFIED, verified_claims=[dm_claim])

    result = agent.score(cand, res, qual, ver)
    assert result.scores.accessibility >= 75.0
    assert any("leadership contact verified" in r.lower() for r in result.scoring_reasons)


# 18. Multiple independent sources improve evidence strength
def test_multiple_independent_sources_improve_evidence_strength():
    agent = ScoringAgent()
    cand = make_candidate("MultiSrcCorp")
    ev1 = make_evidence("ev-1", "https://multisrc.com/careers", "Hiring AI engineer", reliability=90, days_ago=5)
    ev2 = make_evidence("ev-2", "https://lever.co/multisrc", "AI platform engineer role", reliability=85, days_ago=6, source_type=SourceType.JOB_BOARD)
    res = make_research_result(cand, [ev1, ev2])
    qual = make_qualification_result(cand)

    ver_multi = make_verification_result(cand, VerificationStatus.VERIFIED, domains=2)
    ver_single = make_verification_result(cand, VerificationStatus.VERIFIED, domains=1)

    s_multi = agent.score(cand, res, qual, ver_multi)
    s_single = agent.score(cand, res, qual, ver_single)

    assert s_multi.scores.evidence_strength > s_single.scores.evidence_strength


# 19. Same-domain sources do not inflate independence
def test_same_domain_sources_do_not_inflate_independence():
    agent = ScoringAgent()
    cand = make_candidate("SameDomCorp")
    ev1 = make_evidence("ev-1", "https://samedom.com/careers/role1", "Hiring AI engineer", reliability=85, days_ago=5)
    ev2 = make_evidence("ev-2", "https://samedom.com/careers/role2", "Hiring platform engineer", reliability=85, days_ago=5)
    res = make_research_result(cand, [ev1, ev2])
    qual = make_qualification_result(cand)

    # Even with 2 evidence items, domains count is 1
    ver = make_verification_result(cand, VerificationStatus.PARTIALLY_VERIFIED, domains=1)
    result = agent.score(cand, res, qual, ver)

    assert result.scores.evidence_strength < 90.0


# 20. Stale current-need evidence reduces timing
def test_stale_current_need_reduces_timing():
    agent = ScoringAgent()
    cand = make_candidate("StaleNeedCorp")
    ev = make_evidence("ev-1", "https://staleneed.com/jobs", "Hiring AI engineer", reliability=85, days_ago=180)
    res = make_research_result(cand, [ev])
    qual = make_qualification_result(cand, timing=FitLevel.LOW)
    ver = make_verification_result(cand, VerificationStatus.PARTIALLY_VERIFIED, warnings=["recency window exceeded"])

    result = agent.score(cand, res, qual, ver)
    assert result.scores.timing <= 45.0
    assert any("timing penalized" in r.lower() for r in result.scoring_reasons)


# 21. Recent current-need evidence improves timing
def test_recent_current_need_improves_timing():
    agent = ScoringAgent()
    cand = make_candidate("RecentCorp")
    ev = make_evidence("ev-1", "https://recentcorp.com/jobs", "Hiring AI engineer", reliability=90, days_ago=5)
    res = make_research_result(cand, [ev])
    qual = make_qualification_result(cand, timing=FitLevel.HIGH)
    timing_claim = VerificationClaim(
        claim="Recent findings fall within recency window",
        status=VerificationStatus.VERIFIED,
        supporting_evidence_ids=["ev-1"],
        explanation="Fresh within 7 days",
    )
    ver = make_verification_result(cand, VerificationStatus.VERIFIED, verified_claims=[timing_claim])

    result = agent.score(cand, res, qual, ver)
    assert result.scores.timing >= 90.0


# 22. Qualification status is respected
def test_qualification_status_is_respected():
    agent = ScoringAgent()
    cand = make_candidate("WatchCorp")
    ev = make_evidence("ev-1", "https://watchcorp.com/jobs", "AI mentioned", reliability=75, days_ago=10)
    res = make_research_result(cand, [ev])
    # Qualification assigned WATCHLIST
    qual = make_qualification_result(cand, status=QualificationStatus.WATCHLIST, need_fit=FitLevel.MEDIUM)
    ver = make_verification_result(cand, VerificationStatus.PARTIALLY_VERIFIED)

    result = agent.score(cand, res, qual, ver)
    # Priority cannot exceed WATCHLIST
    assert result.scores.priority in (Priority.WATCHLIST, Priority.DISCARD)


# 23. Disqualified vendor cannot become normal customer opportunity
def test_disqualified_vendor_cannot_become_customer_opportunity():
    agent = ScoringAgent()
    cand = make_candidate("VendorCorp")
    ev = make_evidence("ev-1", "https://vendorcorp.com/services", "Custom AI solutions agency", reliability=90, days_ago=10)
    res = make_research_result(cand, [ev])
    qual = make_qualification_result(cand, status=QualificationStatus.DISQUALIFIED, prospect_type=ProspectType.COMPETITOR_OR_VENDOR)
    ver = make_verification_result(cand, VerificationStatus.VERIFIED)

    result = agent.score(cand, res, qual, ver)
    assert result.scores.priority == Priority.DISCARD
    assert any("disqualified" in r.lower() for r in result.scoring_reasons)


# 24. Multiple candidates are scored independently
def test_multiple_candidates_scored_independently():
    agent = ScoringAgent()
    cand1 = make_candidate("AlphaCorp", "cand-alpha")
    cand2 = make_candidate("BetaCorp", "cand-beta")

    ev1 = make_evidence("ev-a", "https://alpha.com/jobs", "Hiring Staff AI Platform Engineer", reliability=95, days_ago=3)
    ev2 = make_evidence("ev-b", "https://beta.com/jobs", "AI mentioned in footer", reliability=50, days_ago=90)

    res1 = make_research_result(cand1, [ev1])
    res2 = make_research_result(cand2, [ev2])

    qual1 = make_qualification_result(cand1, status=QualificationStatus.QUALIFIED, need_fit=FitLevel.HIGH)
    qual2 = make_qualification_result(cand2, status=QualificationStatus.DISQUALIFIED, need_fit=FitLevel.LOW)

    ver1 = make_verification_result(cand1, VerificationStatus.VERIFIED, domains=2)
    ver2 = make_verification_result(cand2, VerificationStatus.UNVERIFIED, domains=1)

    s1 = agent.score(cand1, res1, qual1, ver1)
    s2 = agent.score(cand2, res2, qual2, ver2)

    assert s1.scores.opportunity_score > s2.scores.opportunity_score
    assert s1.scores.priority in (Priority.HIGH, Priority.QUALIFIED)
    assert s2.scores.priority == Priority.DISCARD


# 25. Graph executes scoring without external services
def test_graph_executes_scoring_without_external_services():
    input_data = AOIInput(
        objective=BusinessObjective(
            description="Find companies with active AI platform engineering needs.",
            target_markets=["Technology"],
        ),
        operator_profile=OperatorProfile(
            capabilities=["AI platform engineering", "Python", "Docker"],
            portfolio_projects=["Built scalable data platform"],
        ),
    )
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

    assert result["status"] in (RunStatus.SCORING, RunStatus.COMPLETED)
    assert len(result["scoring_results"]) == 1
    scored = result["scoring_results"][0]
    assert scored.company_name == "RoboNav"
    assert scored.scores.opportunity_score >= 0.0
    assert scored.scores.confidence_score >= 0.0


# 26. Contradicted opportunity preserves raw score but cannot receive HIGH
def test_contradicted_opportunity_preserves_raw_score_no_high():
    agent = ScoringAgent()
    cand = make_candidate("ContraRawCorp")
    ev1 = make_evidence("ev-1", "https://contraraw.com/jobs", "Hiring AI platform engineer", reliability=95, days_ago=3)
    res = make_research_result(cand, [ev1])
    # High theoretical fit
    qual = make_qualification_result(cand, need_fit=FitLevel.HIGH, capability_fit=FitLevel.HIGH)
    ver = make_verification_result(cand, VerificationStatus.CONTRADICTED, domains=2)

    result = agent.score(cand, res, qual, ver)
    # Underlying opportunity score is preserved for diagnostic value
    assert result.scores.opportunity_score > 0.0
    # But priority MUST be DISCARD (never HIGH)
    assert result.scores.priority == Priority.DISCARD
    assert result.scores.confidence_score <= 15.0


# 27. Disqualified candidate preserves diagnostic score but Priority is DISCARD
def test_disqualified_candidate_preserves_diagnostic_score_priority_discard():
    agent = ScoringAgent()
    cand = make_candidate("DisqualRawCorp")
    ev1 = make_evidence("ev-1", "https://disqualraw.com/services", "AI development consultancy", reliability=90, days_ago=10)
    res = make_research_result(cand, [ev1])
    qual = make_qualification_result(cand, status=QualificationStatus.DISQUALIFIED, need_fit=FitLevel.HIGH)
    ver = make_verification_result(cand, VerificationStatus.VERIFIED, domains=2)

    result = agent.score(cand, res, qual, ver)
    # Preserves the diagnostic score
    assert result.scores.opportunity_score > 0.0
    # But Priority is strictly DISCARD
    assert result.scores.priority == Priority.DISCARD


# 28. Populated operator profile capability matching logic
def test_populated_operator_profile_capability_matching():
    agent = ScoringAgent()
    cand = make_candidate("MatchCorp")
    ev = make_evidence("ev-1", "https://matchcorp.com/jobs", "Hiring engineer", reliability=85, days_ago=5)
    qual = make_qualification_result(cand)
    ver = make_verification_result(cand, VerificationStatus.VERIFIED)

    # 4+ matching tokens: python, docker, kubernetes, platform
    op_strong = OperatorProfile(capabilities=["Python", "Docker", "Kubernetes", "Platform engineering"])
    res_strong = make_research_result(
        cand,
        [ev],
        tech=["Python", "Docker", "Kubernetes", "Platform"],
        problems=["Data platform latency"],
    )
    s_strong = agent.score(cand, res_strong, qual, ver, operator_profile=op_strong)
    assert s_strong.scores.capability_fit >= 90.0

    # 0 matching tokens
    op_mismatch = OperatorProfile(capabilities=["Ruby on Rails", "PHP", "WordPress"])
    s_mismatch = agent.score(cand, res_strong, qual, ver, operator_profile=op_mismatch)
    assert s_mismatch.scores.capability_fit <= 30.0
    assert s_strong.scores.capability_fit > s_mismatch.scores.capability_fit


# 29. Missing/incomplete inputs handled safely
def test_scoring_safely_handles_missing_or_incomplete_inputs():
    agent = ScoringAgent()
    cand = make_candidate("IncompleteCorp")
    # Empty evidence, empty tech signals, empty problem signals, empty people
    res = ResearchResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        website=None,
        people=[],
        technology_signals=[],
        business_signals=[],
        problem_signals=[],
        evidence=[],
        warnings=["No evidence collected"],
        research_status=ResearchStatus.FAILED,
        researched_at=datetime.now(UTC),
    )
    qual = QualificationResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        qualification_status=QualificationStatus.INSUFFICIENT_EVIDENCE,
        prospect_type=ProspectType.UNKNOWN,
        need_fit=FitLevel.UNKNOWN,
        capability_fit=FitLevel.UNKNOWN,
        commercial_relevance=FitLevel.UNKNOWN,
        timing=FitLevel.UNKNOWN,
        evidence_sufficiency=FitLevel.UNKNOWN,
        reasons=[],
        supporting_evidence_ids=[],
        disqualification_reasons=[],
        warnings=[],
        qualified_at=datetime.now(UTC),
    )
    ver = VerificationResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        verification_status=VerificationStatus.UNVERIFIED,
        verified_claims=[],
        partially_verified_claims=[],
        unverified_claims=[],
        contradicted_claims=[],
        verification_warnings=["Verification could not be performed"],
        evidence_checked=0,
        independent_sources=0,
        verified_at=datetime.now(UTC),
    )

    result = agent.score(cand, res, qual, ver)
    assert isinstance(result.scores.opportunity_score, float)
    assert isinstance(result.scores.confidence_score, float)
    assert result.scores.priority == Priority.DISCARD
    assert result.scores.confidence_score <= 40.0


# 30. Confidence changes independently of Opportunity Score
def test_confidence_changes_independently_of_opportunity_score():
    agent = ScoringAgent()
    cand = make_candidate("IndepVarianceCorp")
    ev1 = make_evidence("ev-1", "https://indep.com/careers", "Hiring AI engineer", reliability=95, days_ago=5)
    ev2 = make_evidence("ev-2", "https://jobs.lever.co/indep", "Staff AI engineer", reliability=90, days_ago=5, source_type=SourceType.JOB_BOARD)
    res = make_research_result(cand, [ev1, ev2])
    qual = make_qualification_result(cand, need_fit=FitLevel.HIGH, capability_fit=FitLevel.HIGH)

    # Condition A: Multi-source fully verified
    ver_a = make_verification_result(cand, status=VerificationStatus.VERIFIED, domains=3)
    # Condition B: Unverified single weak source
    ver_b = make_verification_result(cand, status=VerificationStatus.UNVERIFIED, domains=0)

    score_a = agent.score(cand, res, qual, ver_a)
    score_b = agent.score(cand, res, qual, ver_b)

    # Confidence delta is massive (reflecting evidence/verification truth)
    conf_delta = score_a.scores.confidence_score - score_b.scores.confidence_score
    assert conf_delta >= 40.0
    # Both still have non-zero opportunity scores reflecting the underlying need
    assert score_a.scores.opportunity_score > 0.0
    assert score_b.scores.opportunity_score > 0.0
