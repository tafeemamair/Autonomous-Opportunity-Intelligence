from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from ..agents.discovery import DiscoveryAgent
from ..agents.qualification import QualificationAgent
from ..agents.research import ResearchAgent
from ..agents.verification import VerificationAgent
from ..schemas.common import RunStatus
from ..schemas.objective import AOIInput
from .state import AOIState

discovery_agent = DiscoveryAgent()
research_agent = ResearchAgent()
qualification_agent = QualificationAgent()
verification_agent = VerificationAgent()


def discovery_node(state: AOIState) -> dict:
    plan = discovery_agent.plan(state.input)
    return {"status": RunStatus.DISCOVERING, "discovery_plan": plan}


def research_node(state: AOIState) -> dict:
    if not state.candidates:
        return {}
    results = []
    for candidate in state.candidates:
        res = research_agent.research(candidate)
        results.append(res)
    return {"status": RunStatus.RESEARCHING, "research_results": results}


def qualification_node(state: AOIState) -> dict:
    if not state.research_results:
        return {}
    results = []
    candidate_by_id = {c.candidate_id: c for c in state.candidates}
    for res in state.research_results:
        candidate = candidate_by_id.get(res.candidate_id)
        if candidate:
            qual = qualification_agent.qualify(
                candidate=candidate,
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


def build_graph():
    builder = StateGraph(AOIState)
    builder.add_node("discovery", discovery_node)
    builder.add_node("research", research_node)
    builder.add_node("qualification", qualification_node)
    builder.add_node("verification", verification_node)
    builder.add_edge(START, "discovery")
    builder.add_edge("discovery", "research")
    builder.add_edge("research", "qualification")
    builder.add_edge("qualification", "verification")
    builder.add_edge("verification", END)
    return builder.compile()


aoi_graph = build_graph()


def create_initial_state(aoi_input: AOIInput) -> AOIState:
    return AOIState(run_id=f"AOI-{uuid4().hex[:12]}", input=aoi_input, status=RunStatus.CREATED)
