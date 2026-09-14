from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from pydantic import HttpUrl, ValidationError

from aoi import (
    AOIReport,
    ReportBuilder,
    ReportEvidence,
    ReportOpportunity,
    ReportStatistics,
    ReportSummary,
)
from aoi.graph.workflow import aoi_graph, create_initial_state, reporting_node
from aoi.schemas.common import Evidence, Priority, RunStatus, Source, SourceType, VerificationStatus
from aoi.schemas.discovery import Candidate
from aoi.schemas.objective import AOIInput, BusinessObjective
from aoi.schemas.opportunity import DecisionMaker
from aoi.schemas.qualification import (
    FitLevel,
    ProspectType,
    QualificationResult,
    QualificationStatus,
)
from aoi.schemas.research import ResearchResult, ResearchStatus
from aoi.schemas.scoring import ScoreBreakdown, ScoringResult
from aoi.schemas.verification import VerificationClaim, VerificationResult


def make_evidence(
    ev_id: str = "ev-1",
    url: str = "https://example.com/source",
    claim: str = "Verified need for AI infrastructure",
    reliability: int = 85,
    recency: int = 90,
    source_type: SourceType = SourceType.COMPANY_WEBSITE,
    status: VerificationStatus = VerificationStatus.VERIFIED,
) -> Evidence:
    now = datetime.now(UTC)
    return Evidence(
        id=ev_id,
        claim=claim,
        source=Source(
            url=HttpUrl(url),
            source_type=source_type,
            title="Source Document",
            publisher="example.com",
            published_at=now - timedelta(days=5),
            accessed_at=now,
            reliability=reliability,
        ),
        evidence_summary=claim,
        verification_status=status,
        recency_score=recency,
    )


def make_candidate(name: str = "AlphaCorp", cand_id: str = "cand-1") -> Candidate:
    clean = "".join(c for c in name.lower() if c.isalnum())
    return Candidate(
        candidate_id=cand_id,
        company_name=name,
        website=HttpUrl(f"https://www.{clean}.com"),
        discovery_strategy="hiring_signal",
        discovery_reason="AI engineer opening",
        discovered_at=datetime.now(UTC),
    )


def make_research_result(
    cand: Candidate,
    evidence: list[Evidence] | None = None,
    problems: list[str] | None = None,
    warnings: list[str] | None = None,
) -> ResearchResult:
    return ResearchResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        website=cand.website,
        people=[DecisionMaker(name="Jane Doe", role="VP Engineering")],
        technology_signals=["Python", "Kubernetes"],
        business_signals=["Expanding AI operations"],
        problem_signals=problems or ["Manual data ingestion bottlenecks"],
        evidence=evidence or [make_evidence(f"ev-{cand.candidate_id}")],
        warnings=warnings or [],
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )


def make_qualification_result(
    cand: Candidate,
    status: QualificationStatus = QualificationStatus.QUALIFIED,
    prospect_type: ProspectType = ProspectType.POTENTIAL_CUSTOMER,
    reasons: list[str] | None = None,
    warnings: list[str] | None = None,
) -> QualificationResult:
    return QualificationResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        qualification_status=status,
        prospect_type=prospect_type,
        need_fit=FitLevel.HIGH,
        capability_fit=FitLevel.HIGH,
        commercial_relevance=FitLevel.HIGH,
        timing=FitLevel.HIGH,
        evidence_sufficiency=FitLevel.HIGH,
        reasons=reasons or ["Strong AI infrastructure requirement"],
        supporting_evidence_ids=[],
        disqualification_reasons=["Disqualified reason"] if status == QualificationStatus.DISQUALIFIED else [],
        warnings=warnings or [],
        qualified_at=datetime.now(UTC),
    )


def make_verification_result(
    cand: Candidate,
    status: VerificationStatus = VerificationStatus.VERIFIED,
    warnings: list[str] | None = None,
) -> VerificationResult:
    claim = VerificationClaim(
        claim=f"{cand.company_name} requires AI workflow automation.",
        status=status,
        supporting_evidence_ids=[f"ev-{cand.candidate_id}"],
        explanation="Validated against authoritative source.",
    )
    return VerificationResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        verification_status=status,
        verified_claims=[claim] if status == VerificationStatus.VERIFIED else [],
        partially_verified_claims=[claim] if status == VerificationStatus.PARTIALLY_VERIFIED else [],
        unverified_claims=[claim] if status == VerificationStatus.UNVERIFIED else [],
        contradicted_claims=[claim] if status == VerificationStatus.CONTRADICTED else [],
        verification_warnings=warnings or [],
        evidence_checked=1,
        independent_sources=2,
        verified_at=datetime.now(UTC),
    )


def make_scoring_result(
    candidate_id: str,
    company_name: str,
    priority: Priority = Priority.HIGH,
    opp_score: float = 85.0,
    conf_score: float = 80.0,
    verification_status: VerificationStatus = VerificationStatus.VERIFIED,
    reasons: list[str] | None = None,
    warnings: list[str] | None = None,
) -> ScoringResult:
    breakdown = ScoreBreakdown(
        need_fit=opp_score,
        capability_fit=opp_score,
        evidence_strength=conf_score,
        timing=75.0,
        commercial_potential=80.0,
        accessibility=70.0,
        opportunity_score=opp_score,
        confidence_score=conf_score,
        priority=priority,
    )
    return ScoringResult(
        candidate_id=candidate_id,
        company_name=company_name,
        scores=breakdown,
        scoring_reasons=reasons or [f"Scored {priority.value} with strong fit"],
        scoring_warnings=warnings or [],
        verification_status=verification_status,
        scored_at=datetime.now(UTC),
    )


# =========================================================================
# 1. SCHEMA VALIDATION TESTS
# =========================================================================


def test_schema_valid_report_evidence():
    ev = ReportEvidence(
        evidence_id="ev-123",
        claim="Company is migrating to LLM orchestration",
        source="https://example.com/job/123",
        source_type=SourceType.OFFICIAL_JOB,
        published_at=datetime.now(UTC),
        accessed_at=datetime.now(UTC),
        evidence_summary="Job posting confirms migration",
        reliability=90.0,
        recency=85.0,
        verification_status="VERIFIED",
    )
    assert ev.evidence_id == "ev-123"
    assert ev.source == "https://example.com/job/123"
    assert ev.reliability == 90.0


def test_schema_valid_report_opportunity():
    breakdown = ScoreBreakdown(
        need_fit=80.0,
        capability_fit=80.0,
        evidence_strength=80.0,
        timing=80.0,
        commercial_potential=80.0,
        accessibility=80.0,
        opportunity_score=80.0,
        confidence_score=80.0,
        priority=Priority.HIGH,
    )
    opp = ReportOpportunity(
        rank=1,
        candidate_id="cand-1",
        company_name="Acme Corp",
        website="https://acme.com",
        opportunity_summary="Acme Corp identified as potential customer.",
        problem="Data ingestion latency",
        potential_solution="Pipeline automation",
        prospect_type=ProspectType.POTENTIAL_CUSTOMER,
        qualification_status=QualificationStatus.QUALIFIED,
        verification_status=VerificationStatus.VERIFIED,
        opportunity_score=80.0,
        confidence_score=80.0,
        priority=Priority.HIGH,
        score_breakdown=breakdown,
        evidence=[],
        scoring_reasons=["High fit"],
        warnings=[],
        recommended_action="Review this opportunity first.",
    )
    assert opp.rank == 1
    assert opp.priority == Priority.HIGH


def test_schema_valid_report_statistics():
    stats = ReportStatistics(
        candidates_discovered=10,
        candidates_researched=8,
        candidates_qualified=6,
        candidates_verified=5,
        opportunities_scored=5,
        high_priority=2,
        qualified_priority=2,
        watchlist_priority=1,
        discard_priority=0,
    )
    assert stats.candidates_discovered == 10
    assert stats.high_priority == 2


def test_schema_valid_report_summary():
    summary = ReportSummary(
        headline="Identified 2 actionable opportunities.",
        overview="Evaluation complete.",
        top_opportunity_count=2,
        high_confidence_count=2,
        warning_count=0,
    )
    assert summary.top_opportunity_count == 2
    assert summary.warning_count == 0


def test_schema_valid_aoi_report():
    summary = ReportSummary(
        headline="Headline",
        overview="Overview",
        top_opportunity_count=0,
        high_confidence_count=0,
        warning_count=0,
    )
    stats = ReportStatistics(
        candidates_discovered=0,
        candidates_researched=0,
        candidates_qualified=0,
        candidates_verified=0,
        opportunities_scored=0,
        high_priority=0,
        qualified_priority=0,
        watchlist_priority=0,
        discard_priority=0,
    )
    report = AOIReport(
        run_id="run-1",
        objective={"description": "Test objective"},
        run_status=RunStatus.COMPLETED,
        generated_at=datetime.now(UTC),
        summary=summary,
        statistics=stats,
        opportunities=[],
        watchlist=[],
        discarded_count=0,
        warnings=[],
    )
    assert report.run_id == "run-1"
    assert report.run_status == RunStatus.COMPLETED


def test_schema_invalid_required_fields():
    with pytest.raises(ValidationError):
        ReportEvidence(evidence_id="ev-1")  # missing required fields


def test_schema_forbid_extra_fields():
    with pytest.raises(ValidationError):
        ReportSummary(
            headline="Headline",
            overview="Overview",
            top_opportunity_count=0,
            high_confidence_count=0,
            warning_count=0,
            unexpected_field="disallowed",
        )


# =========================================================================
# 2. RANKING AND GLOBAL TIE-BREAK TESTS
# =========================================================================


def test_ranking_high_before_qualified():
    builder = ReportBuilder()
    s_qual = make_scoring_result("c-qual", "QualCorp", priority=Priority.QUALIFIED, opp_score=70.0)
    s_high = make_scoring_result("c-high", "HighCorp", priority=Priority.HIGH, opp_score=85.0)

    report = builder.build(scoring_results=[s_qual, s_high])
    assert len(report.opportunities) == 2
    assert report.opportunities[0].candidate_id == "c-high"
    assert report.opportunities[0].rank == 1
    assert report.opportunities[1].candidate_id == "c-qual"
    assert report.opportunities[1].rank == 2


def test_ranking_qualified_before_watchlist():
    builder = ReportBuilder()
    s_watch = make_scoring_result("c-watch", "WatchCorp", priority=Priority.WATCHLIST, opp_score=55.0)
    s_qual = make_scoring_result("c-qual", "QualCorp", priority=Priority.QUALIFIED, opp_score=70.0)

    report = builder.build(scoring_results=[s_watch, s_qual])
    assert len(report.opportunities) == 1
    assert report.opportunities[0].candidate_id == "c-qual"
    assert report.opportunities[0].rank == 1

    assert len(report.watchlist) == 1
    assert report.watchlist[0].candidate_id == "c-watch"
    assert report.watchlist[0].rank == 2


def test_ranking_watchlist_before_discard_and_no_active_rank_for_discard():
    builder = ReportBuilder()
    s_disc = make_scoring_result("c-disc", "DiscCorp", priority=Priority.DISCARD, opp_score=30.0)
    s_watch = make_scoring_result("c-watch", "WatchCorp", priority=Priority.WATCHLIST, opp_score=55.0)

    report = builder.build(scoring_results=[s_disc, s_watch])
    assert len(report.opportunities) == 0
    assert len(report.watchlist) == 1
    assert report.watchlist[0].candidate_id == "c-watch"
    assert report.watchlist[0].rank == 1
    assert report.discarded_count == 1


def test_ranking_score_descending_within_priority():
    builder = ReportBuilder()
    s1 = make_scoring_result("c1", "CorpOne", priority=Priority.HIGH, opp_score=82.0)
    s2 = make_scoring_result("c2", "CorpTwo", priority=Priority.HIGH, opp_score=92.0)

    report = builder.build(scoring_results=[s1, s2])
    assert report.opportunities[0].candidate_id == "c2"
    assert report.opportunities[0].opportunity_score == 92.0
    assert report.opportunities[1].candidate_id == "c1"
    assert report.opportunities[1].opportunity_score == 82.0


def test_ranking_confidence_descending_tie_break():
    builder = ReportBuilder()
    s1 = make_scoring_result("c1", "CorpOne", priority=Priority.HIGH, opp_score=85.0, conf_score=60.0)
    s2 = make_scoring_result("c2", "CorpTwo", priority=Priority.HIGH, opp_score=85.0, conf_score=90.0)

    report = builder.build(scoring_results=[s1, s2])
    assert report.opportunities[0].candidate_id == "c2"
    assert report.opportunities[0].confidence_score == 90.0
    assert report.opportunities[1].candidate_id == "c1"


def test_ranking_company_name_tie_break():
    builder = ReportBuilder()
    s_zeta = make_scoring_result("c1", "ZetaCorp", priority=Priority.HIGH, opp_score=85.0, conf_score=80.0)
    s_alpha = make_scoring_result("c2", "AlphaCorp", priority=Priority.HIGH, opp_score=85.0, conf_score=80.0)

    report = builder.build(scoring_results=[s_zeta, s_alpha])
    assert report.opportunities[0].company_name == "AlphaCorp"
    assert report.opportunities[1].company_name == "ZetaCorp"


def test_ranking_candidate_id_final_tie_break():
    builder = ReportBuilder()
    s_b = make_scoring_result("cand-b", "SameName", priority=Priority.HIGH, opp_score=85.0, conf_score=80.0)
    s_a = make_scoring_result("cand-a", "SameName", priority=Priority.HIGH, opp_score=85.0, conf_score=80.0)

    report = builder.build(scoring_results=[s_b, s_a])
    assert report.opportunities[0].candidate_id == "cand-a"
    assert report.opportunities[1].candidate_id == "cand-b"


def test_ranking_global_rank_across_watchlist():
    """Verify ranks are global: 1 HIGH, 2 QUALIFIED, 3 WATCHLIST (not restarted at 1 for watchlist)."""
    builder = ReportBuilder()
    s_high = make_scoring_result("c-high", "HighCorp", priority=Priority.HIGH, opp_score=88.0)
    s_qual = make_scoring_result("c-qual", "QualCorp", priority=Priority.QUALIFIED, opp_score=72.0)
    s_watch = make_scoring_result("c-watch", "WatchCorp", priority=Priority.WATCHLIST, opp_score=52.0)

    report = builder.build(scoring_results=[s_high, s_qual, s_watch])
    assert len(report.opportunities) == 2
    assert report.opportunities[0].rank == 1
    assert report.opportunities[1].rank == 2

    assert len(report.watchlist) == 1
    assert report.watchlist[0].rank == 3


# =========================================================================
# 3. DATA PRESERVATION TESTS
# =========================================================================


def test_data_preservation_opportunity_and_confidence_score():
    builder = ReportBuilder()
    s = make_scoring_result("c1", "Corp", opp_score=84.3, conf_score=76.8)
    report = builder.build(scoring_results=[s])

    opp = report.opportunities[0]
    assert opp.opportunity_score == 84.3
    assert opp.confidence_score == 76.8


def test_data_preservation_priority_and_score_breakdown():
    builder = ReportBuilder()
    s = make_scoring_result("c1", "Corp", priority=Priority.QUALIFIED, opp_score=68.5)
    report = builder.build(scoring_results=[s])

    opp = report.opportunities[0]
    assert opp.priority == Priority.QUALIFIED
    assert opp.score_breakdown.opportunity_score == 68.5
    assert opp.score_breakdown.priority == Priority.QUALIFIED


def test_data_preservation_verification_and_qualification_status():
    builder = ReportBuilder()
    c = make_candidate("PreserveCorp", "cand-p")
    q = make_qualification_result(c, status=QualificationStatus.QUALIFIED, prospect_type=ProspectType.POTENTIAL_PARTNER)
    v = make_verification_result(c, status=VerificationStatus.PARTIALLY_VERIFIED)
    s = make_scoring_result("cand-p", "PreserveCorp", verification_status=VerificationStatus.PARTIALLY_VERIFIED)

    report = builder.build(
        candidates=[c],
        qualification_results=[q],
        verification_results=[v],
        scoring_results=[s],
    )
    opp = report.opportunities[0]
    assert opp.qualification_status == QualificationStatus.QUALIFIED
    assert opp.prospect_type == ProspectType.POTENTIAL_PARTNER
    assert opp.verification_status == VerificationStatus.PARTIALLY_VERIFIED


def test_data_preservation_evidence_ids_and_fields():
    builder = ReportBuilder()
    c = make_candidate("EvCorp", "cand-e")
    ev = make_evidence("ev-999", url="https://custom.com/ev", claim="Custom claim", reliability=78, recency=65)
    r = make_research_result(c, evidence=[ev])
    s = make_scoring_result("cand-e", "EvCorp")

    report = builder.build(candidates=[c], research_results=[r], scoring_results=[s])
    opp = report.opportunities[0]
    assert len(opp.evidence) == 1
    rep_ev = opp.evidence[0]
    assert rep_ev.evidence_id == "ev-999"
    assert rep_ev.source == "https://custom.com/ev"
    assert rep_ev.claim == "Custom claim"
    assert rep_ev.reliability == 78.0
    assert rep_ev.recency == 65.0
    assert rep_ev.verification_status == "VERIFIED"


# =========================================================================
# 4. SAFETY & INVARIANTS TESTS
# =========================================================================


def test_safety_no_score_recalculation():
    """Report Builder must preserve arbitrary upstream scores without recomputing."""
    builder = ReportBuilder()
    breakdown = ScoreBreakdown(
        need_fit=10.0,
        capability_fit=10.0,
        evidence_strength=10.0,
        timing=10.0,
        commercial_potential=10.0,
        accessibility=10.0,
        opportunity_score=99.9,  # Intentionally non-standard
        confidence_score=88.8,
        priority=Priority.HIGH,
    )
    s = ScoringResult(
        candidate_id="c-raw",
        company_name="RawCorp",
        scores=breakdown,
        scoring_reasons=["Manual override"],
        scoring_warnings=[],
        verification_status=VerificationStatus.VERIFIED,
        scored_at=datetime.now(UTC),
    )
    report = builder.build(scoring_results=[s])
    assert report.opportunities[0].opportunity_score == 99.9
    assert report.opportunities[0].confidence_score == 88.8
    assert report.opportunities[0].priority == Priority.HIGH


def test_safety_no_fabricated_evidence():
    """Evidence in ReportOpportunity must belong strictly to that candidate."""
    builder = ReportBuilder()
    c1 = make_candidate("Corp1", "c1")
    c2 = make_candidate("Corp2", "c2")
    r1 = make_research_result(c1, evidence=[make_evidence("ev-c1")])
    r2 = make_research_result(c2, evidence=[make_evidence("ev-c2")])
    s1 = make_scoring_result("c1", "Corp1")
    s2 = make_scoring_result("c2", "Corp2")

    report = builder.build(
        candidates=[c1, c2],
        research_results=[r1, r2],
        scoring_results=[s1, s2],
    )
    opp1 = next(o for o in report.opportunities if o.candidate_id == "c1")
    assert all(e.evidence_id == "ev-c1" for e in opp1.evidence)


def test_safety_no_fabricated_decision_makers_or_claims():
    """Missing upstream fields must be represented safely as None, never fabricated."""
    builder = ReportBuilder()
    s = make_scoring_result("c-bare", "BareCorp")
    # No candidate, research, qualification, or verification passed
    report = builder.build(scoring_results=[s])
    opp = report.opportunities[0]
    assert opp.website is None
    assert opp.problem is None
    assert opp.potential_solution is None
    assert opp.evidence == []


def test_safety_missing_data_safe():
    """Builder runs safely when pipeline contains empty or partial components."""
    builder = ReportBuilder()
    report = builder.build(
        run_id="run-partial",
        objective={"description": "Test"},
        candidates=[],
        research_results=[],
        qualification_results=[],
        verification_results=[],
        scoring_results=[],
    )
    assert report.opportunities == []
    assert report.watchlist == []
    assert report.discarded_count == 0


def test_safety_contradiction_preserved():
    """Contradicted verification status generates conflict resolution recommendation."""
    builder = ReportBuilder()
    c = make_candidate("ContradictedCorp", "cand-x")
    v = make_verification_result(c, status=VerificationStatus.CONTRADICTED)
    s = make_scoring_result("cand-x", "ContradictedCorp", verification_status=VerificationStatus.CONTRADICTED, priority=Priority.HIGH)

    report = builder.build(candidates=[c], verification_results=[v], scoring_results=[s])
    opp = report.opportunities[0]
    assert opp.recommended_action == "Resolve the conflicting evidence before considering this opportunity."


def test_safety_disqualified_candidates_counted_as_discard():
    """Disqualified candidates must be counted as discard and not in active opportunities."""
    builder = ReportBuilder()
    c = make_candidate("DisqualifiedCorp", "cand-d")
    q = make_qualification_result(c, status=QualificationStatus.DISQUALIFIED)
    s = make_scoring_result("cand-d", "DisqualifiedCorp", priority=Priority.DISCARD, opp_score=20.0)

    report = builder.build(
        candidates=[c],
        qualification_results=[q],
        scoring_results=[s],
    )
    assert report.opportunities == []
    assert report.watchlist == []
    assert report.discarded_count == 1
    assert report.statistics.discard_priority == 1


def test_safety_watchlist_separated_correctly():
    """Watchlist opportunities must appear only in report.watchlist."""
    builder = ReportBuilder()
    s_watch = make_scoring_result("c-w", "WatchCorp", priority=Priority.WATCHLIST, opp_score=55.0)
    report = builder.build(scoring_results=[s_watch])

    assert report.opportunities == []
    assert len(report.watchlist) == 1
    assert report.watchlist[0].company_name == "WatchCorp"
    assert report.watchlist[0].recommended_action == "Monitor the identified signal and reassess if stronger evidence of an active need appears."


# =========================================================================
# 5. SUMMARY & STATISTICS TESTS
# =========================================================================


def test_statistics_count_correctly():
    builder = ReportBuilder()
    c1 = make_candidate("C1", "c1")
    c2 = make_candidate("C2", "c2")
    c3 = make_candidate("C3", "c3")
    c4 = make_candidate("C4", "c4")

    s1 = make_scoring_result("c1", "C1", priority=Priority.HIGH)
    s2 = make_scoring_result("c2", "C2", priority=Priority.QUALIFIED)
    s3 = make_scoring_result("c3", "C3", priority=Priority.WATCHLIST)
    s4 = make_scoring_result("c4", "C4", priority=Priority.DISCARD)

    report = builder.build(
        candidates=[c1, c2, c3, c4],
        research_results=[make_research_result(c1), make_research_result(c2)],
        qualification_results=[make_qualification_result(c1)],
        verification_results=[make_verification_result(c1)],
        scoring_results=[s1, s2, s3, s4],
    )

    assert report.statistics.candidates_discovered == 4
    assert report.statistics.candidates_researched == 2
    assert report.statistics.candidates_qualified == 1
    assert report.statistics.candidates_verified == 1
    assert report.statistics.opportunities_scored == 4
    assert report.statistics.high_priority == 1
    assert report.statistics.qualified_priority == 1
    assert report.statistics.watchlist_priority == 1
    assert report.statistics.discard_priority == 1


def test_deterministic_summary():
    builder = ReportBuilder()
    c1 = make_candidate("Alpha", "c1")
    s1 = make_scoring_result("c1", "Alpha", priority=Priority.HIGH, opp_score=90.0, conf_score=85.0)

    report = builder.build(candidates=[c1], scoring_results=[s1])
    assert report.summary.top_opportunity_count == 1
    assert report.summary.high_confidence_count == 1
    assert "Identified 1 actionable opportunity" in report.summary.headline
    assert "1 High Priority" in report.summary.overview


def test_warning_aggregation_and_deduplication():
    """Builder aggregates warnings from all stages and deduplicates them in stable order."""
    builder = ReportBuilder()
    c = make_candidate("WarnCorp", "c1")
    r = make_research_result(c, warnings=["Common warning", "Research warning"])
    q = make_qualification_result(c, warnings=["Common warning", "Qual warning"])
    v = make_verification_result(c, warnings=["Verification warning"])
    s = make_scoring_result("c1", "WarnCorp", warnings=["Scoring warning", "Common warning"])

    report = builder.build(
        candidates=[c],
        research_results=[r],
        qualification_results=[q],
        verification_results=[v],
        scoring_results=[s],
        warnings=["Pipeline warning", "Common warning"],
    )

    expected = [
        "Pipeline warning",
        "Common warning",
        "Research warning",
        "Qual warning",
        "Verification warning",
        "Scoring warning",
    ]
    assert report.warnings == expected
    assert report.summary.warning_count == len(expected)


# =========================================================================
# 6. EDGE CASES
# =========================================================================


def test_edge_case_empty_run():
    builder = ReportBuilder()
    report = builder.build(
        run_id="run-empty",
        objective={"description": "Find AI companies"},
        scoring_results=[],
    )
    assert report.opportunities == []
    assert report.watchlist == []
    assert report.discarded_count == 0
    assert report.summary.top_opportunity_count == 0
    assert report.summary.headline == "No actionable opportunities identified."
    assert "none met the threshold" in report.summary.overview


def test_edge_case_partial_run():
    builder = ReportBuilder()
    report = builder.build(
        status=RunStatus.PARTIAL,
        scoring_results=[make_scoring_result("c1", "Corp1", priority=Priority.HIGH)],
    )
    assert report.run_status == RunStatus.PARTIAL


def test_edge_case_deterministic_repeated_generation():
    """Identical state must produce identical report content excluding generated_at."""
    builder = ReportBuilder()
    c1 = make_candidate("CorpA", "c1")
    c2 = make_candidate("CorpB", "c2")
    s1 = make_scoring_result("c1", "CorpA", priority=Priority.HIGH, opp_score=85.0)
    s2 = make_scoring_result("c2", "CorpB", priority=Priority.QUALIFIED, opp_score=70.0)

    rep1 = builder.build(candidates=[c1, c2], scoring_results=[s1, s2])
    rep2 = builder.build(candidates=[c1, c2], scoring_results=[s1, s2])

    dump1 = rep1.model_dump(exclude={"generated_at"})
    dump2 = rep2.model_dump(exclude={"generated_at"})
    assert dump1 == dump2


# =========================================================================
# 7. WORKFLOW & GRAPH INTEGRATION TESTS
# =========================================================================


def test_graph_reporting_node_executes():
    input_data = AOIInput(objective=BusinessObjective(description="Find companies with AI automation needs."))
    state = create_initial_state(input_data)
    state.candidates = [make_candidate("NodeCorp", "cand-n")]
    state.scoring_results = [make_scoring_result("cand-n", "NodeCorp", priority=Priority.HIGH)]

    update = reporting_node(state)
    assert update["status"] == RunStatus.COMPLETED
    assert isinstance(update["report"], AOIReport)
    assert len(update["report"].opportunities) == 1
    assert update["report"].opportunities[0].company_name == "NodeCorp"


def test_graph_aoistate_report_populated():
    input_data = AOIInput(objective=BusinessObjective(description="Find companies with AI automation needs."))
    state = create_initial_state(input_data)
    result = aoi_graph.invoke(state)

    assert result["status"] == RunStatus.COMPLETED
    assert "report" in result
    assert isinstance(result["report"], AOIReport)
    assert result["report"].run_status == RunStatus.COMPLETED


def test_graph_reporting_no_external_services():
    """Ensure reporting node performs zero HTTP, network, or external API calls."""
    builder = ReportBuilder()
    s = make_scoring_result("c1", "NetCorp", priority=Priority.HIGH)

    with patch("httpx.Client.request") as mock_request, patch("httpx.AsyncClient.request") as mock_async:
        report = builder.build(scoring_results=[s])
        assert mock_request.call_count == 0
        assert mock_async.call_count == 0
        assert len(report.opportunities) == 1


# =========================================================================
# 8. AUDIT CONFIRMATION TESTS
# =========================================================================


def test_high_confidence_count_based_on_confidence_score_at_least_80():
    """Confirm high_confidence_count is strictly based on confidence_score >= 80.0, NOT opportunity_score."""
    builder = ReportBuilder()
    # High opp score (95.0), but low confidence score (79.9) -> NOT high confidence
    s_low_conf = make_scoring_result("c1", "LowConf", priority=Priority.HIGH, opp_score=95.0, conf_score=79.9)
    # Lower opp score (65.0), but high confidence score (80.0) -> IS high confidence
    s_high_conf = make_scoring_result("c2", "HighConf", priority=Priority.QUALIFIED, opp_score=65.0, conf_score=80.0)
    # High opp score (90.0), high confidence score (90.0) -> IS high confidence
    s_both_high = make_scoring_result("c3", "BothHigh", priority=Priority.HIGH, opp_score=90.0, conf_score=90.0)

    report = builder.build(scoring_results=[s_low_conf, s_high_conf, s_both_high])
    assert report.summary.high_confidence_count == 2


def test_reporting_node_completed_with_warnings_only_remains_completed():
    input_data = AOIInput(objective=BusinessObjective(description="Find companies with AI automation needs."))
    state = create_initial_state(input_data)
    state.status = RunStatus.SCORING
    state.warnings = ["Test warning 1", "Test warning 2"]

    update = reporting_node(state)
    assert update["status"] == RunStatus.COMPLETED
    assert update["report"].run_status == RunStatus.COMPLETED
    assert len(update["report"].warnings) >= 2


def test_reporting_node_existing_partial_status_remains_partial():
    input_data = AOIInput(objective=BusinessObjective(description="Find companies with AI automation needs."))
    state = create_initial_state(input_data)
    state.status = RunStatus.PARTIAL

    update = reporting_node(state)
    assert update["status"] == RunStatus.PARTIAL
    assert update["report"].run_status == RunStatus.PARTIAL


def test_reporting_node_failed_and_cancelled_preserved():
    input_data = AOIInput(objective=BusinessObjective(description="Find companies with AI automation needs."))
    state_failed = create_initial_state(input_data)
    state_failed.status = RunStatus.FAILED
    update_failed = reporting_node(state_failed)
    assert update_failed["status"] == RunStatus.FAILED
    assert update_failed["report"].run_status == RunStatus.FAILED

    state_cancelled = create_initial_state(input_data)
    state_cancelled.status = RunStatus.CANCELLED
    update_cancelled = reporting_node(state_cancelled)
    assert update_cancelled["status"] == RunStatus.CANCELLED
    assert update_cancelled["report"].run_status == RunStatus.CANCELLED


def test_reporting_node_errors_and_warnings_alone_do_not_force_partial():
    input_data = AOIInput(objective=BusinessObjective(description="Find companies with AI automation needs."))
    state = create_initial_state(input_data)
    state.status = RunStatus.SCORING
    state.errors = ["Recovered non-fatal error"]
    state.warnings = ["Non-fatal warning"]

    update = reporting_node(state)
    assert update["status"] == RunStatus.COMPLETED
    assert update["report"].run_status == RunStatus.COMPLETED

