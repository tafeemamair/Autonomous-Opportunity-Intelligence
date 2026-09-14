from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from ..agents.discovery import DiscoveryAgent
from ..agents.qualification import QualificationAgent
from ..agents.research import ResearchAgent
from ..agents.scoring import ScoringAgent
from ..agents.verification import VerificationAgent
from ..report import ReportBuilder
from ..schemas.common import RunStatus
from ..schemas.objective import AOIInput
from .state import AOIState

discovery_agent = DiscoveryAgent()
research_agent = ResearchAgent()
qualification_agent = QualificationAgent()
verification_agent = VerificationAgent()
scoring_agent = ScoringAgent()
report_builder = ReportBuilder()


def discovery_node(state: AOIState) -> dict:
    plan = discovery_agent.plan(state.input)
    return {"status": RunStatus.DISCOVERING, "discovery_plan": plan}


def research_node(state: AOIState) -> dict:
    if not state.candidates:
        return {}
    results = []
    for cand in state.candidates:
        res = research_agent.research(cand)
        results.append(res)
    return {"status": RunStatus.RESEARCHING, "research_results": results}


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
    builder.add_node("reporting", reporting_node)
    builder.add_edge(START, "discovery")
    builder.add_edge("discovery", "research")
    builder.add_edge("research", "qualification")
    builder.add_edge("qualification", "verification")
    builder.add_edge("verification", "scoring")
    builder.add_edge("scoring", "reporting")
    builder.add_edge("reporting", END)
    return builder.compile()



aoi_graph = build_graph()


def create_initial_state(aoi_input: AOIInput) -> AOIState:
    return AOIState(run_id=f"AOI-{uuid4().hex[:12]}", input=aoi_input, status=RunStatus.CREATED)
