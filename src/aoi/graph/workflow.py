from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from ..agents.discovery import DiscoveryAgent
from ..schemas.common import RunStatus
from ..schemas.objective import AOIInput
from .state import AOIState

discovery_agent = DiscoveryAgent()


def discovery_node(state: AOIState) -> dict:
    plan = discovery_agent.plan(state.input)
    return {"status": RunStatus.DISCOVERING, "discovery_plan": plan}


def build_graph():
    builder = StateGraph(AOIState)
    builder.add_node("discovery", discovery_node)
    builder.add_edge(START, "discovery")
    builder.add_edge("discovery", END)
    return builder.compile()


aoi_graph = build_graph()


def create_initial_state(aoi_input: AOIInput) -> AOIState:
    return AOIState(run_id=f"AOI-{uuid4().hex[:12]}", input=aoi_input, status=RunStatus.CREATED)
