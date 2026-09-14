from datetime import UTC, datetime, timedelta

from pydantic import HttpUrl

from aoi.agents.verification import VerificationAgent
from aoi.graph.state import RunStatus
from aoi.graph.workflow import aoi_graph
from aoi.schemas.common import Evidence, Source, SourceType, VerificationStatus
from aoi.schemas.discovery import Candidate
from aoi.schemas.objective import BusinessObjective, Constraints, OperatorProfile
from aoi.schemas.opportunity import DecisionMaker
from aoi.schemas.qualification import (
    FitLevel,
    ProspectType,
    QualificationResult,
    QualificationStatus,
)
from aoi.schemas.research import ResearchResult, ResearchStatus


def make_evidence(
    ev_id: str,
    url: str,
    claim: str,
    summary: str = "",
    source_type: SourceType = SourceType.COMPANY_WEBSITE,
    reliability: int = 85,
    days_ago: int = 10,
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
        evidence_summary=summary or claim,
        verification_status=VerificationStatus.UNVERIFIED,
        recency_score=max(0, 100 - days_ago),
    )


def make_candidate(name: str = "TestCorp", cand_id: str = "cand-1") -> Candidate:
    clean_name = "".join(c for c in name.lower() if c.isalnum())
    return Candidate(
        candidate_id=cand_id,
        company_name=name,
        website=HttpUrl(f"https://www.{clean_name}.com"),
        discovery_strategy="hiring_signal",
        discovery_reason="AI infrastructure opportunity",
        discovered_at=datetime.now(UTC),
    )


def make_research_result(
    cand: Candidate,
    evidence: list[Evidence],
    people: list[DecisionMaker] | None = None,
    tech: list[str] | None = None,
    business: list[str] | None = None,
    problems: list[str] | None = None,
) -> ResearchResult:
    return ResearchResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        website=cand.website,
        people=people or [],
        technology_signals=tech or ["Python", "Docker"],
        business_signals=business or ["Expanding engineering"],
        problem_signals=problems or ["Hiring bottleneck"],
        evidence=evidence,
        research_status=ResearchStatus.COMPLETED,
        researched_at=datetime.now(UTC),
    )


def make_qualification_result(
    cand: Candidate,
    status: QualificationStatus = QualificationStatus.QUALIFIED,
    prospect_type: ProspectType = ProspectType.POTENTIAL_CUSTOMER,
    evidence_ids: list[str] | None = None,
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
        reasons=["Active hiring for AI platform engineer"],
        supporting_evidence_ids=evidence_ids or [],
        disqualification_reasons=[],
        warnings=[],
        qualified_at=datetime.now(UTC),
    )


# 1. Two independent strong sources -> VERIFIED
def test_two_independent_strong_sources_verifies_need():
    agent = VerificationAgent()
    cand = make_candidate("Bedrock Ocean Exploration")
    ev1 = make_evidence(
        "ev-1",
        "https://bedrockocean.com/careers",
        "Hiring Staff AI Platform Engineer",
        "Job posting looking for AI platform engineer",
        source_type=SourceType.COMPANY_WEBSITE,
        reliability=90,
        days_ago=5,
    )
    ev2 = make_evidence(
        "ev-2",
        "https://jobs.lever.co/bedrockocean/ai-engineer",
        "Staff AI Platform Engineer job description",
        "Open role for AI engineer at Bedrock",
        source_type=SourceType.JOB_BOARD,
        reliability=80,
        days_ago=7,
    )
    res = make_research_result(cand, [ev1, ev2])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-1", "ev-2"])

    v_res = agent.verify(cand, res, qual)
    assert v_res.verification_status == VerificationStatus.VERIFIED
    assert len(v_res.verified_claims) >= 1
    assert any("need" in cl.claim.lower() or "active ai engineering" in cl.claim.lower() for cl in v_res.verified_claims)


# 2. One strong source -> PARTIALLY_VERIFIED
def test_one_strong_source_yields_partially_verified():
    agent = VerificationAgent()
    cand = make_candidate("SoloCorp")
    ev1 = make_evidence(
        "ev-1",
        "https://solocorp.com/jobs",
        "Hiring AI Platform Engineer",
        source_type=SourceType.COMPANY_WEBSITE,
        reliability=90,
        days_ago=10,
    )
    res = make_research_result(cand, [ev1])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-1"])

    v_res = agent.verify(cand, res, qual)
    assert v_res.verification_status == VerificationStatus.PARTIALLY_VERIFIED
    assert any(cl.status == VerificationStatus.PARTIALLY_VERIFIED for cl in v_res.partially_verified_claims)


# 3. One weak source -> UNVERIFIED
def test_one_weak_source_yields_unverified():
    agent = VerificationAgent()
    cand = make_candidate("WeakCorp")
    ev1 = make_evidence(
        "ev-1",
        "https://randomdirectory.com/listing/weakcorp",
        "Hiring AI Engineer maybe",
        source_type=SourceType.DIRECTORY,
        reliability=40,
        days_ago=10,
    )
    res = make_research_result(cand, [ev1])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-1"])

    v_res = agent.verify(cand, res, qual)
    assert v_res.verification_status == VerificationStatus.UNVERIFIED
    assert any(cl.status == VerificationStatus.UNVERIFIED for cl in v_res.unverified_claims)


# 4. Same-domain sources count as one independent source
def test_same_domain_counts_as_single_independent_source():
    agent = VerificationAgent()
    cand = make_candidate("SameDomainCorp")
    ev1 = make_evidence(
        "ev-1",
        "https://samedomain.com/careers/role1",
        "Hiring AI Platform Engineer",
        days_ago=5,
    )
    ev2 = make_evidence(
        "ev-2",
        "https://samedomain.com/about/hiring",
        "We are hiring AI engineers",
        days_ago=5,
    )
    res = make_research_result(cand, [ev1, ev2])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-1", "ev-2"])

    v_res = agent.verify(cand, res, qual)
    assert v_res.independent_sources == 1
    assert v_res.verification_status == VerificationStatus.PARTIALLY_VERIFIED


# 5. Duplicate/same-source evidence does not produce corroboration
def test_duplicate_source_evidence_no_corroboration():
    agent = VerificationAgent()
    cand = make_candidate("DupeCorp")
    ev1 = make_evidence(
        "ev-1",
        "https://dupecorp.com/careers",
        "Hiring AI Engineer",
        days_ago=5,
    )
    ev2 = make_evidence(
        "ev-2",
        "https://dupecorp.com/careers",
        "Hiring AI Engineer duplicate posting",
        days_ago=5,
    )
    res = make_research_result(cand, [ev1, ev2])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-1"])

    v_res = agent.verify(cand, res, qual)
    need_claims = [c for c in v_res.partially_verified_claims if "active ai engineering" in c.claim.lower()]
    assert len(need_claims) == 1
    assert need_claims[0].status == VerificationStatus.PARTIALLY_VERIFIED
    assert len(need_claims[0].corroborating_evidence_ids) == 0


# 6. Contradictory reliable evidence -> CONTRADICTED
def test_contradictory_reliable_evidence_marks_contradicted():
    agent = VerificationAgent()
    cand = make_candidate("FreezeCorp")
    ev1 = make_evidence(
        "ev-1",
        "https://freezecorp.com/jobs",
        "Hiring AI Platform Engineer",
        source_type=SourceType.COMPANY_WEBSITE,
        days_ago=5,
    )
    ev2 = make_evidence(
        "ev-2",
        "https://news.ycombinator.com/freezecorp",
        "Company announced a hiring freeze and cancelled role yesterday",
        source_type=SourceType.NEWS,
        reliability=80,
        days_ago=1,
    )
    res = make_research_result(cand, [ev1, ev2])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-1"])

    v_res = agent.verify(cand, res, qual)
    assert v_res.verification_status == VerificationStatus.CONTRADICTED
    assert len(v_res.contradicted_claims) >= 1
    assert any(cl.status == VerificationStatus.CONTRADICTED for cl in v_res.contradicted_claims)


# 7. Old evidence cannot independently verify a current need (recency rule)
def test_old_evidence_cannot_verify_current_need():
    agent = VerificationAgent()
    cand = make_candidate("StaleCorp")
    ev1 = make_evidence(
        "ev-1",
        "https://stalecorp.com/careers",
        "Hiring AI Platform Engineer",
        days_ago=180,
    )
    ev2 = make_evidence(
        "ev-2",
        "https://lever.co/stalecorp/job",
        "AI platform engineer role",
        source_type=SourceType.JOB_BOARD,
        days_ago=200,
    )
    res = make_research_result(cand, [ev1, ev2])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-1", "ev-2"])
    constraints = Constraints(recency_days=90)

    v_res = agent.verify(cand, res, qual, constraints=constraints)
    need_claim = next((c for c in v_res.partially_verified_claims if "active ai engineering" in c.claim.lower()), None)
    assert need_claim is not None
    assert any("stale" in w.lower() or "older than 90 days" in w.lower() or "recency window" in w.lower() for w in v_res.verification_warnings)


# 8. Recent evidence preferred for current claims, historical tech allowed
def test_recent_need_with_historical_tech():
    agent = VerificationAgent()
    cand = make_candidate("HistTechCorp")
    ev_need1 = make_evidence(
        "ev-1",
        "https://histcorp.com/careers",
        "Hiring AI Engineer",
        days_ago=15,
    )
    ev_need2 = make_evidence(
        "ev-2",
        "https://greenhouse.io/histcorp",
        "Hiring AI Platform Engineer",
        source_type=SourceType.JOB_BOARD,
        days_ago=10,
    )
    ev_tech = make_evidence(
        "ev-3",
        "https://histcorp.com/blog/2023-tech-stack",
        "Built our infrastructure using Python and Docker autonomous agents",
        days_ago=300,
    )
    res = make_research_result(cand, [ev_need1, ev_need2, ev_tech], tech=["Python", "Docker"])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-1", "ev-2"])

    v_res = agent.verify(cand, res, qual)
    assert any(cl.status == VerificationStatus.VERIFIED and "active ai engineering" in cl.claim.lower() for cl in v_res.verified_claims)
    assert any(cl.status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED) and "technology" in cl.claim.lower() for cl in (v_res.verified_claims + v_res.partially_verified_claims))
    assert v_res.verification_status == VerificationStatus.VERIFIED


# 9. Evidence IDs in VerificationClaim are valid
def test_evidence_ids_in_claim_are_valid():
    agent = VerificationAgent()
    cand = make_candidate("IdCheckCorp")
    ev1 = make_evidence("ev-real-1", "https://idcorp.com/jobs", "Hiring AI Engineer", days_ago=5)
    res = make_research_result(cand, [ev1])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-real-1"])

    v_res = agent.verify(cand, res, qual)
    valid_ids = {ev1.id}
    all_claims = v_res.verified_claims + v_res.partially_verified_claims + v_res.unverified_claims + v_res.contradicted_claims
    for cl in all_claims:
        for eid in cl.supporting_evidence_ids:
            assert eid in valid_ids
        for eid in cl.corroborating_evidence_ids:
            assert eid in valid_ids
        for eid in cl.contradiction_evidence_ids:
            assert eid in valid_ids


# 10. Fabricated evidence IDs are never emitted
def test_fabricated_evidence_ids_never_emitted():
    agent = VerificationAgent()
    cand = make_candidate("NoFakeCorp")
    ev1 = make_evidence("ev-101", "https://nofake.com/careers", "Hiring AI Engineer", days_ago=5)
    res = make_research_result(cand, [ev1])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-fake-999"])

    v_res = agent.verify(cand, res, qual)
    all_claims = v_res.verified_claims + v_res.partially_verified_claims + v_res.unverified_claims + v_res.contradicted_claims
    for cl in all_claims:
        assert "ev-fake-999" not in cl.supporting_evidence_ids
        assert "ev-fake-999" not in cl.corroborating_evidence_ids
        assert "ev-fake-999" not in cl.contradiction_evidence_ids


# 11. Qualification claims are actually evaluated
def test_qualification_claims_are_evaluated():
    agent = VerificationAgent()
    cand = make_candidate("EvalCorp")
    ev1 = make_evidence("ev-1", "https://evalcorp.com/jobs", "Hiring Staff AI Platform Engineer", days_ago=5)
    res = make_research_result(cand, [ev1])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-1"])

    v_res = agent.verify(cand, res, qual)
    claim_types = [c.claim for c in (v_res.verified_claims + v_res.partially_verified_claims + v_res.unverified_claims)]
    assert any("engineering requirement" in c.lower() for c in claim_types)
    assert any("classified as" in c.lower() for c in claim_types)
    assert any("technology" in c.lower() for c in claim_types)


# 12. Unsupported qualification claims remain UNVERIFIED
def test_unsupported_qualification_claim_unverified():
    agent = VerificationAgent()
    cand = make_candidate("EmptyClaimCorp")
    res = make_research_result(cand, [])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, [])

    v_res = agent.verify(cand, res, qual)
    assert v_res.verification_status == VerificationStatus.UNVERIFIED
    assert any(cl.status == VerificationStatus.UNVERIFIED for cl in v_res.unverified_claims)


# 13. Decision-maker verification (corroborated vs unverified)
def test_decision_maker_verification():
    agent = VerificationAgent()
    cand = make_candidate("LeaderCorp")
    p1 = DecisionMaker(name="Alice Smith", role="VP of Engineering")
    p2 = DecisionMaker(name="Bob Jones", role="CTO")
    ev1 = make_evidence("ev-1", "https://leadercorp.com/team", "Alice Smith is VP of Engineering at LeaderCorp", days_ago=10)
    ev2 = make_evidence("ev-2", "https://news.com/leadercorp-alice", "Alice Smith joined LeaderCorp as VP of Engineering", source_type=SourceType.NEWS, days_ago=12)
    res = make_research_result(cand, [ev1, ev2], people=[p1, p2])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-1"])

    v_res = agent.verify(cand, res, qual)
    alice_claim = next((c for c in v_res.verified_claims if "Alice Smith" in c.claim), None)
    assert alice_claim is not None
    assert alice_claim.status == VerificationStatus.VERIFIED

    bob_claim = next((c for c in v_res.unverified_claims if "Bob Jones" in c.claim), None)
    assert bob_claim is not None
    assert bob_claim.status == VerificationStatus.UNVERIFIED


# 14. Competitor/vendor classification verified
def test_vendor_classification_verified():
    agent = VerificationAgent()
    cand = make_candidate("DevShop Inc")
    ev1 = make_evidence("ev-1", "https://devshop.com/services", "Custom AI solutions and AI development company", days_ago=10)
    ev2 = make_evidence("ev-2", "https://clutch.co/profile/devshop", "Software development agency offering custom AI development", source_type=SourceType.DIRECTORY, days_ago=15)
    res = make_research_result(cand, [ev1, ev2])
    qual = make_qualification_result(cand, QualificationStatus.DISQUALIFIED, ProspectType.COMPETITOR_OR_VENDOR, ["ev-1", "ev-2"])

    v_res = agent.verify(cand, res, qual)
    prospect_claim = next((c for c in v_res.verified_claims if "classified as competitor_or_vendor" in c.claim.lower()), None)
    assert prospect_claim is not None
    assert prospect_claim.status == VerificationStatus.VERIFIED
    assert v_res.verification_status == VerificationStatus.VERIFIED


# 15. Potential partner classification verified
def test_partner_classification_verified():
    agent = VerificationAgent()
    cand = make_candidate("PartnerCorp")
    ev1 = make_evidence("ev-1", "https://partnercorp.com/partners", "Join our technology partner program", days_ago=10)
    res = make_research_result(cand, [ev1])
    qual = make_qualification_result(cand, QualificationStatus.WATCHLIST, ProspectType.POTENTIAL_PARTNER, ["ev-1"])

    v_res = agent.verify(cand, res, qual)
    prospect_claim = next((c for c in v_res.partially_verified_claims if "classified as potential_partner" in c.claim.lower()), None)
    assert prospect_claim is not None
    assert prospect_claim.status == VerificationStatus.PARTIALLY_VERIFIED


# 16. Missing research handled safely
def test_missing_research_handled_safely():
    agent = VerificationAgent()
    cand = make_candidate("MissingResearchCorp")
    res = ResearchResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        website=cand.website,
        evidence=[],
        research_status=ResearchStatus.FAILED,
        warnings=["Research provider timed out"],
        researched_at=datetime.now(UTC),
    )
    qual = make_qualification_result(cand, QualificationStatus.INSUFFICIENT_EVIDENCE, ProspectType.UNKNOWN, [])

    v_res = agent.verify(cand, res, qual)
    assert v_res.verification_status == VerificationStatus.UNVERIFIED
    assert v_res.evidence_checked == 0
    assert v_res.independent_sources == 0


# 17. Empty qualification results handled safely
def test_empty_qualification_handled_safely():
    agent = VerificationAgent()
    cand = make_candidate("EmptyQualCorp")
    res = make_research_result(cand, [])
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

    v_res = agent.verify(cand, res, qual)
    assert v_res.verification_status == VerificationStatus.UNVERIFIED
    assert len(v_res.verified_claims) == 0


# 18. Multiple candidates verified independently
def test_multiple_candidates_verified_independently():
    agent = VerificationAgent()

    # Candidate A: Strong evidence
    cand_a = make_candidate("Company A", "cand-a")
    ev_a1 = make_evidence("ev-a1", "https://companya.com/careers", "Hiring AI engineer", days_ago=5)
    ev_a2 = make_evidence("ev-a2", "https://indeed.com/companya", "AI engineer open role", source_type=SourceType.JOB_BOARD, days_ago=6)
    res_a = make_research_result(cand_a, [ev_a1, ev_a2])
    qual_a = make_qualification_result(cand_a, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-a1", "ev-a2"])

    # Candidate B: Contradicted
    cand_b = make_candidate("Company B", "cand-b")
    ev_b1 = make_evidence("ev-b1", "https://companyb.com/careers", "Hiring AI engineer", days_ago=5)
    ev_b2 = make_evidence("ev-b2", "https://news.com/companyb", "Company B announced hiring freeze and layoffs", source_type=SourceType.NEWS, days_ago=2)
    res_b = make_research_result(cand_b, [ev_b1, ev_b2])
    qual_b = make_qualification_result(cand_b, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-b1"])

    v_res_a = agent.verify(cand_a, res_a, qual_a)
    v_res_b = agent.verify(cand_b, res_b, qual_b)

    assert v_res_a.verification_status == VerificationStatus.VERIFIED
    assert v_res_b.verification_status == VerificationStatus.CONTRADICTED
    assert v_res_a.candidate_id == "cand-a"
    assert v_res_b.candidate_id == "cand-b"


# 19. Graph executes Verification node
def test_graph_executes_verification_node():
    from aoi.graph.workflow import create_initial_state
    from aoi.schemas.objective import AOIInput

    input_data = AOIInput(
        objective=BusinessObjective(
            description="Find companies with AI platform engineering needs.",
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

    assert result["status"] in (RunStatus.VERIFYING, RunStatus.SCORING)
    assert len(result["verification_results"]) == 1
    assert result["verification_results"][0].company_name == "RoboNav"


# 20. Graph works without external services
def test_graph_works_without_external_services():
    from aoi.graph.workflow import create_initial_state
    from aoi.schemas.objective import AOIInput

    input_data = AOIInput(
        objective=BusinessObjective(
            description="Find companies with AI platform engineering needs.",
        ),
    )
    state = create_initial_state(input_data)
    final_state = aoi_graph.invoke(state)
    assert final_state["verification_results"] == []
    assert final_state["status"] == RunStatus.DISCOVERING


# 21. Bedrock Ocean Exploration canonical acceptance scenario
def test_canonical_bedrock_ocean_acceptance_scenario():
    agent = VerificationAgent()
    cand = Candidate(
        candidate_id="cand-bedrock",
        company_name="Bedrock Ocean Exploration",
        website=HttpUrl("https://www.bedrockocean.com"),
        discovery_strategy="hiring_signal",
        discovery_reason="Autonomous maritime robotics and data platform",
        discovered_at=datetime.now(UTC),
    )
    ev1 = make_evidence(
        "ev-bedrock-1",
        "https://www.bedrockocean.com/careers/staff-ai-platform-engineer",
        "Staff AI Platform Engineer position: Bedrock Ocean Exploration is hiring a Staff AI Platform Engineer to build scalable edge pipelines.",
        summary="Bedrock Ocean Exploration is seeking a Staff AI Platform Engineer for cloud orchestration.",
        source_type=SourceType.COMPANY_WEBSITE,
        reliability=90,
        days_ago=7,
    )
    ev2 = make_evidence(
        "ev-bedrock-2",
        "https://jobs.lever.co/bedrockocean/01a87b64",
        "Job Application for Staff AI Platform Engineer at Bedrock Ocean Exploration",
        summary="Active job listing on Lever for Staff AI Platform Engineer with Python, Kubernetes, and sensor telemetry requirements.",
        source_type=SourceType.JOB_BOARD,
        reliability=85,
        days_ago=9,
    )
    ev3 = make_evidence(
        "ev-bedrock-3",
        "https://techcrunch.com/2023/10/bedrock-ocean-autonomous-survey",
        "Bedrock Ocean closes financing round to scale autonomous ocean floor mapping fleet.",
        summary="TechCrunch report on Bedrock's autonomous subsea vehicle fleet and data infrastructure.",
        source_type=SourceType.NEWS,
        reliability=80,
        days_ago=120,
    )

    p1 = DecisionMaker(name="Anthony DiMare", role="CEO & Co-founder")
    ev_exec = make_evidence(
        "ev-bedrock-4",
        "https://www.bedrockocean.com/about",
        "Anthony DiMare is the CEO of Bedrock Ocean Exploration.",
        source_type=SourceType.COMPANY_WEBSITE,
        reliability=90,
        days_ago=30,
    )

    res = make_research_result(
        cand,
        evidence=[ev1, ev2, ev3, ev_exec],
        people=[p1],
        tech=["Python", "Kubernetes", "Sensor Telemetry"],
        business=["Ocean floor mapping", "Fleet operations"],
        problems=["Edge sensor pipeline throughput"],
    )

    qual = QualificationResult(
        candidate_id=cand.candidate_id,
        company_name=cand.company_name,
        qualification_status=QualificationStatus.QUALIFIED,
        prospect_type=ProspectType.POTENTIAL_CUSTOMER,
        need_fit=FitLevel.HIGH,
        capability_fit=FitLevel.HIGH,
        commercial_relevance=FitLevel.HIGH,
        timing=FitLevel.HIGH,
        evidence_sufficiency=FitLevel.HIGH,
        reasons=["Active hiring for Staff AI Platform Engineer verified across careers portal and ATS"],
        supporting_evidence_ids=["ev-bedrock-1", "ev-bedrock-2"],
        disqualification_reasons=[],
        warnings=[],
        qualified_at=datetime.now(UTC),
    )

    v_res = agent.verify(
        candidate=cand,
        research_result=res,
        qualification_result=qual,
        constraints=Constraints(recency_days=90),
    )

    assert v_res.independent_sources >= 3
    assert v_res.verification_status == VerificationStatus.VERIFIED
    assert len(v_res.verified_claims) >= 2
    assert len(v_res.contradicted_claims) == 0

    dm_claims = [c for c in v_res.verified_claims + v_res.partially_verified_claims if "Anthony DiMare" in c.claim]
    assert len(dm_claims) == 1


# 22. Stale qualification downgraded by Verification
def test_stale_qualification_downgraded_by_verification():
    agent = VerificationAgent()
    cand = make_candidate("StaleQualCorp")
    ev1 = make_evidence("ev-1", "https://stalequal.com/careers", "Hiring AI Engineer", days_ago=120)
    ev2 = make_evidence("ev-2", "https://jobs.lever.co/stalequal", "AI engineer role", source_type=SourceType.JOB_BOARD, days_ago=130)
    res = make_research_result(cand, [ev1, ev2])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-1", "ev-2"])

    v_res = agent.verify(cand, res, qual, constraints=Constraints(recency_days=90))
    assert v_res.verification_status == VerificationStatus.PARTIALLY_VERIFIED
    assert any("older than 90 days" in w or "stale" in w.lower() or "recency window" in w.lower() for w in v_res.verification_warnings)


# 23. Contradiction surfaced rather than hidden
def test_contradiction_surfaced_with_warnings():
    agent = VerificationAgent()
    cand = make_candidate("DownsizeCorp")
    ev1 = make_evidence("ev-1", "https://downsizecorp.com/careers", "Hiring Staff AI Platform Engineer", days_ago=10)
    ev2 = make_evidence("ev-2", "https://wsj.com/downsizecorp", "DownsizeCorp announced layoffs and shut down engineering expansion", source_type=SourceType.NEWS, days_ago=2)
    res = make_research_result(cand, [ev1, ev2])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-1"])

    v_res = agent.verify(cand, res, qual)
    assert v_res.verification_status == VerificationStatus.CONTRADICTED
    assert len(v_res.contradicted_claims) > 0
    contradicted_claim = v_res.contradicted_claims[0]
    assert len(contradicted_claim.contradiction_evidence_ids) > 0
    assert "ev-2" in contradicted_claim.contradiction_evidence_ids


# 24. Full claim traceability and referential integrity audit
def test_full_claim_traceability_and_referential_integrity():
    agent = VerificationAgent()
    cand = make_candidate("TraceCorp")
    ev1 = make_evidence("ev-t1", "https://tracecorp.com/jobs/ai-platform", "Hiring AI platform engineer to automate pipelines", days_ago=10)
    ev2 = make_evidence("ev-t2", "https://techradar.com/tracecorp", "TraceCorp deploys autonomous LLM orchestration", source_type=SourceType.NEWS, days_ago=20)
    dm = DecisionMaker(name="Carol Danvers", role="Head of AI")
    ev3 = make_evidence("ev-t3", "https://tracecorp.com/team", "Carol Danvers is Head of AI at TraceCorp", days_ago=15)

    res = make_research_result(cand, [ev1, ev2, ev3], people=[dm], tech=["LLM orchestration"])
    qual = make_qualification_result(cand, QualificationStatus.QUALIFIED, ProspectType.POTENTIAL_CUSTOMER, ["ev-t1", "ev-t2"])

    v_res = agent.verify(cand, res, qual)
    all_claims = v_res.verified_claims + v_res.partially_verified_claims + v_res.unverified_claims + v_res.contradicted_claims
    valid_ids = {ev1.id, ev2.id, ev3.id}

    assert len(all_claims) >= 4  # need, tech, timing, prospect_type, decision_maker
    for claim in all_claims:
        assert isinstance(claim.status, VerificationStatus)
        assert len(claim.explanation) > 0
        for eid in claim.supporting_evidence_ids:
            assert eid in valid_ids
        for eid in claim.corroborating_evidence_ids:
            assert eid in valid_ids
        for eid in claim.contradiction_evidence_ids:
            assert eid in valid_ids

