import logging
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from ..agents.discovery import DiscoveryAgent
from ..agents.intelligence import IntelligenceQualityAgent
from ..agents.qualification import QualificationAgent
from ..agents.research import ResearchAgent
from ..agents.scoring import ScoringAgent
from ..agents.verification import VerificationAgent
from ..report import ReportBuilder
from ..schemas.common import RunStatus
from ..schemas.discovery import DiscoveryBudget
from ..schemas.objective import AOIInput
from .state import AOIState

logger = logging.getLogger(__name__)

discovery_agent = DiscoveryAgent()
research_agent = ResearchAgent()
qualification_agent = QualificationAgent()
verification_agent = VerificationAgent()
scoring_agent = ScoringAgent()
intelligence_quality_agent = IntelligenceQualityAgent()
report_builder = ReportBuilder()

_discovery_provider = None


def set_discovery_provider(provider) -> None:
    """Explicitly inject a discovery provider (for testing or custom providers)."""
    global _discovery_provider
    _discovery_provider = provider


def get_discovery_provider():
    """Retrieve the explicitly configured discovery provider."""
    global _discovery_provider
    return _discovery_provider


def discovery_node(state: AOIState) -> dict:
    budget = getattr(state.input.constraints, "discovery_budget", None) or DiscoveryBudget()
    plan = discovery_agent.plan(state.input, budget=budget)
    update = {"status": RunStatus.DISCOVERING, "discovery_plan": plan}

    # If candidates were already pre-seeded in state, preserve them and compute discovery evaluation
    if state.candidates:
        eval_res = discovery_agent.evaluate_discovery(
            len(state.candidates), state.candidates, plan=plan
        )
        update["discovery_evaluation"] = eval_res
        return update

    provider = get_discovery_provider()
    if provider is None:
        warning_msg = (
            "Discovery provider configuration unavailable (missing API key or credentials)."
        )
        update["warnings"] = [warning_msg]
        return update

    try:
        try:
            raw_candidates = provider.discover(plan, budget=plan.budget)
        except TypeError:
            raw_candidates = provider.discover(plan)
        if raw_candidates:
            deduped = discovery_agent.deduplicate_candidates(
                raw_candidates, objective=state.input.objective
            )
            prioritized = discovery_agent.prioritize_candidates(
                deduped, objective=state.input.objective
            )
            # Enforce max_candidates cap after deduplication & prioritization
            retained = prioritized[:plan.budget.max_candidates]
            if len(prioritized) > plan.budget.max_candidates:
                msg = f"Discovery budget candidate ceiling reached: retained {len(retained)} of {len(prioritized)} candidates."
                update["warnings"] = list(state.warnings) + [msg]

            eval_res = discovery_agent.evaluate_discovery(
                len(raw_candidates), retained, plan=plan
            )
            update["candidates"] = retained
            update["discovery_evaluation"] = eval_res
        else:
            update["warnings"] = ["Discovery provider executed but returned 0 candidates."]
            update["discovery_evaluation"] = discovery_agent.evaluate_discovery(
                0, [], plan=plan
            )
    except Exception as exc:
        error_msg = f"Discovery execution failed: {exc}"
        logger.warning(error_msg)
        update["warnings"] = [error_msg]
        update["errors"] = [error_msg]
        update["status"] = RunStatus.PARTIAL

    return update


def research_node(state: AOIState) -> dict:
    if not state.candidates:
        return {}
    prioritized_cands = discovery_agent.prioritize_candidates(
        state.candidates, objective=state.input.objective
    )
    results = []
    for cand in prioritized_cands:
        res = research_agent.research(cand)
        results.append(res)
    eval_res = research_agent.evaluate_research(results)
    return {
        "status": RunStatus.RESEARCHING,
        "research_results": results,
        "research_evaluation": eval_res,
    }


def qualification_node(state: AOIState) -> dict:
    if not state.research_results:
        return {}
    results = []
    candidate_by_id = {c.candidate_id: c for c in state.candidates}
    for res in state.research_results:
        cand = candidate_by_id.get(res.candidate_id)
        if cand:
            qual = qualification_agent.qualify(
                candidate=cand,
                research_result=res,
                objective=state.input.objective,
                operator_profile=state.input.operator_profile,
                constraints=state.input.constraints,
            )
            results.append(qual)
    return {"status": RunStatus.QUALIFYING, "qualification_results": results}


def verification_node(state: AOIState) -> dict:
    if not state.research_results or not state.qualification_results:
        return {}
    results = []
    candidate_by_id = {c.candidate_id: c for c in state.candidates}
    research_by_id = {r.candidate_id: r for r in state.research_results}
    for qual in state.qualification_results:
        candidate = candidate_by_id.get(qual.candidate_id)
        research = research_by_id.get(qual.candidate_id)
        if candidate and research:
            ver = verification_agent.verify(
                candidate=candidate,
                research_result=research,
                qualification_result=qual,
                objective=state.input.objective,
                constraints=state.input.constraints,
            )
            results.append(ver)
    return {"status": RunStatus.VERIFYING, "verification_results": results}


def scoring_node(state: AOIState) -> dict:
    if not state.verification_results:
        return {}
    results = []
    candidate_by_id = {c.candidate_id: c for c in state.candidates}
    research_by_id = {r.candidate_id: r for r in state.research_results}
    qualification_by_id = {q.candidate_id: q for q in state.qualification_results}

    for ver in state.verification_results:
        candidate = candidate_by_id.get(ver.candidate_id)
        research = research_by_id.get(ver.candidate_id)
        qual = qualification_by_id.get(ver.candidate_id)
        if candidate and research and qual:
            scored = scoring_agent.score(
                candidate=candidate,
                research_result=research,
                qualification_result=qual,
                verification_result=ver,
                objective=state.input.objective,
                operator_profile=state.input.operator_profile,
                constraints=state.input.constraints,
            )
            results.append(scored)
    return {"status": RunStatus.SCORING, "scoring_results": results}


def intelligence_quality_node(state: AOIState) -> dict:
    if not state.scoring_results:
        return {}
    results = []
    candidate_by_id = {c.candidate_id: c for c in state.candidates}
    research_by_id = {r.candidate_id: r for r in state.research_results}
    qualification_by_id = {q.candidate_id: q for q in state.qualification_results}
    verification_by_id = {v.candidate_id: v for v in state.verification_results}

    operator_profile = state.input.operator_profile if state.input else None
    constraints = state.input.constraints if state.input else None
    objective = state.input.objective if state.input else None

    for scored in state.scoring_results:
        candidate = candidate_by_id.get(scored.candidate_id)
        research = research_by_id.get(scored.candidate_id)
        qual = qualification_by_id.get(scored.candidate_id)
        ver = verification_by_id.get(scored.candidate_id)

        quality_res = intelligence_quality_agent.evaluate(
            candidate=candidate,
            research_result=research,
            qualification_result=qual,
            verification_result=ver,
            scoring_result=scored,
            objective=objective,
            operator_profile=operator_profile,
            constraints=constraints,
            run_warnings=state.warnings,
        )
        results.append(quality_res)

    return {"quality_results": results}


def reporting_node(state: AOIState) -> dict:
    prior_status = state.status
    if prior_status in (RunStatus.FAILED, RunStatus.CANCELLED):
        final_status = prior_status
    elif prior_status == RunStatus.PARTIAL:
        state.status = RunStatus.REPORTING
        final_status = RunStatus.PARTIAL
    else:
        state.status = RunStatus.REPORTING
        final_status = RunStatus.COMPLETED

    report = report_builder.build(state, status=final_status)
    return {"status": final_status, "report": report}


def build_graph():
    builder = StateGraph(AOIState)
    builder.add_node("discovery", discovery_node)
    builder.add_node("research", research_node)
    builder.add_node("qualification", qualification_node)
    builder.add_node("verification", verification_node)
    builder.add_node("scoring", scoring_node)
    builder.add_node("intelligence_quality", intelligence_quality_node)
    builder.add_node("reporting", reporting_node)
    builder.add_edge(START, "discovery")
    builder.add_edge("discovery", "research")
    builder.add_edge("research", "qualification")
    builder.add_edge("qualification", "verification")
    builder.add_edge("verification", "scoring")
    builder.add_edge("scoring", "intelligence_quality")
    builder.add_edge("intelligence_quality", "reporting")
    builder.add_edge("reporting", END)
    return builder.compile()


aoi_graph = build_graph()


def create_initial_state(aoi_input: AOIInput) -> AOIState:
    return AOIState(run_id=f"AOI-{uuid4().hex[:12]}", input=aoi_input, status=RunStatus.CREATED)
