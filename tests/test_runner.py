import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import HttpUrl
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from aoi.cli import main
from aoi.db.models import Run
from aoi.db.session import init_db, persist_run
from aoi.graph.workflow import set_discovery_provider
from aoi.runner import AOIRunner, ObjectiveValidationError, load_objective
from aoi.schemas.common import RunStatus
from aoi.schemas.discovery import Candidate, DiscoveryPlan
from aoi.schemas.objective import AOIInput, BusinessObjective
from aoi.schemas.report import AOIReport


@pytest.fixture
def temp_obj_file(tmp_path: Path) -> Path:
    """Fixture to create a valid objective JSON file."""
    obj_data = {
        "description": "Find enterprise companies with active AI pipeline orchestration needs.",
        "target_markets": ["North America"],
        "target_opportunities": ["Workflow Automation"],
    }
    file_path = tmp_path / "objective.json"
    file_path.write_text(json.dumps(obj_data), encoding="utf-8")
    return file_path


@pytest.fixture
def temp_full_input_file(tmp_path: Path) -> Path:
    """Fixture to create a valid full AOIInput JSON file."""
    obj_data = {
        "objective": {
            "description": "Find enterprise companies with active AI pipeline orchestration needs.",
            "target_markets": ["North America"],
            "target_opportunities": ["Workflow Automation"],
        },
        "operator_profile": {
            "capabilities": ["Python", "Kubernetes"],
        },
        "constraints": {
            "recency_days": 60,
        },
    }
    file_path = tmp_path / "full_input.json"
    file_path.write_text(json.dumps(obj_data), encoding="utf-8")
    return file_path


@pytest.fixture
def memory_db():
    """In-memory SQLite database for test persistence."""
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    init_db(bind_engine=engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return engine, session_factory


class MockDiscoveryProvider:
    """Deterministic mock discovery provider for automated tests."""

    def __init__(self, candidates: list[Candidate] | None = None):
        self.candidates = candidates or []
        self.calls = 0

    def discover(self, plan: DiscoveryPlan) -> list[Candidate]:
        self.calls += 1
        return self.candidates


# =========================================================================
# 1. OBJECTIVE VALIDATION TESTS
# =========================================================================


def test_load_objective_from_valid_business_objective_file(temp_obj_file: Path):
    aoi_input = load_objective(temp_obj_file)
    assert isinstance(aoi_input, AOIInput)
    assert aoi_input.objective.description.startswith("Find enterprise")
    assert aoi_input.objective.target_markets == ["North America"]


def test_load_objective_from_valid_aoi_input_file(temp_full_input_file: Path):
    aoi_input = load_objective(temp_full_input_file)
    assert isinstance(aoi_input, AOIInput)
    assert "Python" in aoi_input.operator_profile.capabilities
    assert aoi_input.constraints.recency_days == 60


def test_load_objective_from_dict():
    data = {
        "description": "Find enterprise companies with active AI pipeline orchestration needs.",
    }
    aoi_input = load_objective(data)
    assert isinstance(aoi_input, AOIInput)
    assert aoi_input.objective.description == data["description"]


def test_load_objective_from_json_string():
    raw = json.dumps({"description": "Find enterprise companies with active AI automation needs."})
    aoi_input = load_objective(raw)
    assert isinstance(aoi_input, AOIInput)


def test_load_objective_from_instance():
    original = AOIInput(
        objective=BusinessObjective(description="Find companies with active AI needs.")
    )
    result = load_objective(original)
    assert result is original


def test_load_objective_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_objective("non_existent_path_12345.json")


def test_load_objective_invalid_json_syntax(tmp_path: Path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ unquoted_key: invalid }", encoding="utf-8")
    with pytest.raises(ObjectiveValidationError, match="Invalid JSON syntax"):
        load_objective(bad_file)


def test_load_objective_invalid_schema_short_description(tmp_path: Path):
    short_file = tmp_path / "short.json"
    short_file.write_text(json.dumps({"description": "short"}), encoding="utf-8")
    with pytest.raises(ObjectiveValidationError, match="validation failed"):
        load_objective(short_file)


def test_load_objective_invalid_schema_extra_fields(tmp_path: Path):
    extra_file = tmp_path / "extra.json"
    extra_file.write_text(
        json.dumps(
            {
                "description": "Find companies with active AI automation needs.",
                "unsupported_field": "disallowed",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ObjectiveValidationError, match="validation failed"):
        load_objective(extra_file)


def test_load_objective_non_dict_json(tmp_path: Path):
    array_file = tmp_path / "array.json"
    array_file.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ObjectiveValidationError, match="must be a JSON object"):
        load_objective(array_file)


# =========================================================================
# 2. CLI ARGUMENT HANDLING TESTS
# =========================================================================


def test_cli_validate_success(temp_obj_file: Path, capsys):
    exit_code = main(["validate", str(temp_obj_file)])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "[VALID] Objective is valid:" in captured.out


def test_cli_validate_missing_file(capsys):
    exit_code = main(["validate", "non_existent_objective.json"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "[ERROR] Objective file not found" in captured.err


def test_cli_validate_invalid_json(tmp_path: Path, capsys):
    bad_file = tmp_path / "invalid.json"
    bad_file.write_text("{bad", encoding="utf-8")
    exit_code = main(["validate", str(bad_file)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "[ERROR] Objective validation failed" in captured.err


def test_cli_run_missing_file(capsys):
    exit_code = main(["run", "missing_file_xyz.json"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "[ERROR] Objective file not found" in captured.err


def test_cli_run_invalid_json(tmp_path: Path, capsys):
    bad_file = tmp_path / "invalid_json.json"
    bad_file.write_text("{bad", encoding="utf-8")
    exit_code = main(["run", str(bad_file)])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "[ERROR] Invalid objective JSON" in captured.err


def test_cli_no_args_displays_help(capsys):
    exit_code = main([])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower() or "aoi" in captured.out.lower()


def test_cli_unknown_command(capsys):
    exit_code = main(["unknown_subcommand"])
    assert exit_code != 0


# =========================================================================
# 3. RUNNER BEHAVIOR & LIFECYCLE TESTS
# =========================================================================


def test_runner_run_id_generation(temp_obj_file: Path):
    runner = AOIRunner(persist_to_db=False)
    result = runner.run(temp_obj_file)
    assert result.run_id.startswith("AOI-")


def test_runner_initial_state_lifecycle(temp_obj_file: Path):
    runner = AOIRunner(persist_to_db=False)
    result = runner.run(temp_obj_file)
    assert result.state.run_id == result.run_id
    assert result.status in (RunStatus.COMPLETED, RunStatus.PARTIAL)


def test_runner_unpacking_support(temp_obj_file: Path):
    runner = AOIRunner(persist_to_db=False)
    status, report, state = runner.run(temp_obj_file)
    assert isinstance(status, RunStatus)
    assert isinstance(report, AOIReport)


def test_runner_successful_execution(temp_obj_file: Path):
    runner = AOIRunner(persist_to_db=False)
    result = runner.run(temp_obj_file)
    assert result.status == RunStatus.COMPLETED
    assert result.report is not None
    assert result.report.run_status == RunStatus.COMPLETED
    assert isinstance(result.report, AOIReport)


def test_runner_output_file_serialization(temp_obj_file: Path, tmp_path: Path):
    out_path = tmp_path / "reports" / "report.json"
    runner = AOIRunner(persist_to_db=False)
    result = runner.run(temp_obj_file, output_path=out_path)

    assert result.output_file == out_path
    assert out_path.exists()
    content = out_path.read_text(encoding="utf-8")
    assert '"run_id":' in content


def test_aoireport_json_round_trip(temp_obj_file: Path, tmp_path: Path):
    out_path = tmp_path / "round_trip.json"
    runner = AOIRunner(persist_to_db=False)
    result = runner.run(temp_obj_file, output_path=out_path)

    json_text = out_path.read_text(encoding="utf-8")
    loaded_report = AOIReport.model_validate_json(json_text)
    assert loaded_report.run_id == result.report.run_id
    assert loaded_report.run_status == result.report.run_status
    assert (
        loaded_report.summary.top_opportunity_count == result.report.summary.top_opportunity_count
    )


def test_aoireport_json_round_trip_equality(temp_obj_file: Path, tmp_path: Path):
    out_path = tmp_path / "eq.json"
    runner = AOIRunner(persist_to_db=False)
    result = runner.run(temp_obj_file, output_path=out_path)

    json_text = out_path.read_text(encoding="utf-8")
    loaded_report = AOIReport.model_validate_json(json_text)
    assert result.report.model_dump() == loaded_report.model_dump()


def test_runner_partial_execution_preserves_partial(temp_obj_file: Path):
    runner = AOIRunner(persist_to_db=False)
    now_ts = __import__("datetime").datetime.now(__import__("datetime").UTC)
    # Simulate a state with RunStatus.PARTIAL
    with patch("aoi.runner.aoi_graph.invoke") as mock_invoke:
        mock_invoke.return_value = {
            "status": RunStatus.PARTIAL,
            "report": AOIReport(
                run_id="run-p",
                objective={"description": "Test"},
                run_status=RunStatus.PARTIAL,
                generated_at=now_ts,
                summary=__import__("aoi.schemas.report", fromlist=["ReportSummary"]).ReportSummary(
                    headline="Partial run",
                    overview="Partial",
                    top_opportunity_count=0,
                    high_confidence_count=0,
                    warning_count=1,
                ),
                statistics=__import__(
                    "aoi.schemas.report", fromlist=["ReportStatistics"]
                ).ReportStatistics(
                    candidates_discovered=0,
                    candidates_researched=0,
                    candidates_qualified=0,
                    candidates_verified=0,
                    opportunities_scored=0,
                    high_priority=0,
                    qualified_priority=0,
                    watchlist_priority=0,
                    discard_priority=0,
                ),
                opportunities=[],
                watchlist=[],
                discarded_count=0,
                warnings=["Partial pipeline warning"],
            ),
        }
        result = runner.run(temp_obj_file)
        assert result.status == RunStatus.PARTIAL
        assert result.report.run_status == RunStatus.PARTIAL


def test_runner_handles_keyboard_interrupt(temp_obj_file: Path):
    runner = AOIRunner(persist_to_db=False)
    with patch("aoi.runner.aoi_graph.invoke", side_effect=KeyboardInterrupt):
        result = runner.run(temp_obj_file, propagate_interrupt=False)
        assert result.status == RunStatus.CANCELLED
        assert result.report is None


def test_runner_keyboard_interrupt_propagation(temp_obj_file: Path):
    runner = AOIRunner(persist_to_db=False)
    with patch("aoi.runner.aoi_graph.invoke", side_effect=KeyboardInterrupt):
        with pytest.raises(KeyboardInterrupt):
            runner.run(temp_obj_file, propagate_interrupt=True)


def test_runner_handles_graph_exception(temp_obj_file: Path):
    runner = AOIRunner(persist_to_db=False)
    with patch("aoi.runner.aoi_graph.invoke", side_effect=RuntimeError("Fatal pipeline crash")):
        result = runner.run(temp_obj_file)
        assert result.status == RunStatus.FAILED
        assert result.report is None
        assert any("Fatal pipeline crash" in err for err in result.state.errors)


# =========================================================================
# 4. DATABASE PERSISTENCE TESTS
# =========================================================================


def test_runner_database_run_persistence(temp_obj_file: Path, memory_db):
    engine, session_factory = memory_db
    runner = AOIRunner(db_session_factory=session_factory, persist_to_db=True)
    result = runner.run(temp_obj_file)

    with session_factory() as session:
        record = session.get(Run, result.run_id)
        assert record is not None
        assert record.status == RunStatus.COMPLETED
        assert record.objective["description"] is not None
        assert record.started_at is not None
        assert record.completed_at is not None


def test_runner_database_cancellation_persistence(temp_obj_file: Path, memory_db):
    engine, session_factory = memory_db
    runner = AOIRunner(db_session_factory=session_factory, persist_to_db=True)

    with patch("aoi.runner.aoi_graph.invoke", side_effect=KeyboardInterrupt):
        result = runner.run(temp_obj_file, propagate_interrupt=False)
        assert result.status == RunStatus.CANCELLED

    with session_factory() as session:
        record = session.get(Run, result.run_id)
        assert record is not None
        assert record.status == RunStatus.CANCELLED


def test_runner_database_failure_resilience(temp_obj_file: Path):
    """Database persistence failure should not crash the runner or prevent report delivery."""
    mock_factory = MagicMock(side_effect=Exception("Database locked or unreachable"))
    runner = AOIRunner(db_session_factory=mock_factory, persist_to_db=True)

    result = runner.run(temp_obj_file)
    assert result.status == RunStatus.COMPLETED
    assert result.report is not None


def test_db_persist_run_helper_creates_and_updates(memory_db):
    engine, session_factory = memory_db
    persist_run(
        run_id="run-test-1",
        status="CREATED",
        objective={"description": "Test objective"},
        session_factory=session_factory,
    )
    with session_factory() as session:
        r = session.get(Run, "run-test-1")
        assert r is not None
        assert r.status == "CREATED"

    # Update record
    persist_run(
        run_id="run-test-1",
        status="COMPLETED",
        objective={"description": "Test objective"},
        statistics={"candidates_discovered": 5},
        session_factory=session_factory,
    )
    with session_factory() as session:
        r = session.get(Run, "run-test-1")
        assert r.status == "COMPLETED"
        assert r.statistics["candidates_discovered"] == 5


# =========================================================================
# 5. DETERMINISTIC MOCKED END-TO-END TESTS
# =========================================================================


def test_runner_deterministic_mocked_end_to_end(temp_obj_file: Path):
    cand = Candidate(
        candidate_id="cand-mock-1",
        company_name="MockAutomationCorp",
        website=HttpUrl("https://mockautomation.com"),
        discovery_strategy="hiring_signal",
        discovery_reason="Active hiring for AI workflow engineers",
        discovered_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
    )
    mock_provider = MockDiscoveryProvider(candidates=[cand])
    runner = AOIRunner(persist_to_db=False)

    try:
        result = runner.run(temp_obj_file, discovery_provider=mock_provider)
        assert result.status == RunStatus.COMPLETED
        assert result.report is not None
        assert result.report.statistics.candidates_discovered >= 1
        assert mock_provider.calls == 1
    finally:
        set_discovery_provider(None)


def test_runner_no_external_network_during_run(temp_obj_file: Path):
    """Verify runner and graph execute with zero live HTTP/network calls when offline."""
    runner = AOIRunner(persist_to_db=False)
    with (
        patch("httpx.Client.request") as mock_req,
        patch("httpx.AsyncClient.request") as mock_async,
    ):
        result = runner.run(temp_obj_file)
        assert mock_req.call_count == 0
        assert mock_async.call_count == 0
        assert result.status == RunStatus.COMPLETED


def test_cli_run_successful_command_execution(temp_obj_file: Path, tmp_path: Path, capsys):
    out_file = tmp_path / "cli_report.json"
    exit_code = main(["run", str(temp_obj_file), "--output", str(out_file)])
    assert exit_code == 0
    assert out_file.exists()

    captured = capsys.readouterr()
    assert "AOI Run" in captured.out
    assert "✓ Discovery" in captured.out
    assert "Status: COMPLETED" in captured.out


def test_cli_run_keyboard_interrupt_exits_130(temp_obj_file: Path, capsys):
    with patch("aoi.cli.AOIRunner.run", side_effect=KeyboardInterrupt):
        exit_code = main(["run", str(temp_obj_file)])
        assert exit_code == 130
        captured = capsys.readouterr()
        assert "[CANCELLED]" in captured.err


def test_cli_run_failure_exits_1(temp_obj_file: Path, capsys):
    with patch("aoi.cli.AOIRunner.run", side_effect=RuntimeError("Unrecoverable failure")):
        exit_code = main(["run", str(temp_obj_file)])
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "[ERROR]" in captured.err
