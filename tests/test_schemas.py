from aoi.schemas.objective import AOIInput, BusinessObjective


def test_objective_defaults_are_valid():
    payload = AOIInput(objective=BusinessObjective(description="Find companies with credible AI automation needs."))
    assert payload.constraints.require_evidence is True
    assert payload.constraints.require_human_approval is True
    assert payload.objective.minimum_opportunities == 20


def test_extra_fields_are_rejected():
    try:
        AOIInput.model_validate({"objective": {"description": "Find AI automation opportunities.", "unexpected": True}})
    except ValueError:
        return
    raise AssertionError("Unexpected fields should be rejected.")
