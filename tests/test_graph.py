from aoi.graph.workflow import aoi_graph, create_initial_state
from aoi.schemas.common import RunStatus
from aoi.schemas.objective import AOIInput, BusinessObjective


def test_graph_runs_without_external_services():
    input_data = AOIInput(objective=BusinessObjective(description="Find companies with recent AI automation needs."))
    state = create_initial_state(input_data)
    result = aoi_graph.invoke(state)
    assert result["status"] == RunStatus.COMPLETED
    assert result["discovery_plan"] is not None
    assert len(result["discovery_plan"].strategies) == 5
    assert result["report"] is not None

