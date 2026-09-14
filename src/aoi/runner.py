import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from .graph.state import AOIState
from .graph.workflow import aoi_graph, create_initial_state, set_discovery_provider
from .schemas.common import RunStatus
from .schemas.discovery import Candidate
from .schemas.objective import AOIInput, BusinessObjective
from .schemas.report import AOIReport

logger = logging.getLogger(__name__)


class ObjectiveValidationError(ValueError):
    """Raised when an objective input fails schema or file validation."""


@dataclass
class RunResult:
    """Encapsulates the execution result of an AOI pipeline run."""

    run_id: str
    status: RunStatus
    report: AOIReport | None
    state: AOIState
    output_file: Path | None = None

    def __iter__(self):
        """Allow tuple unpacking: status, report, state = result."""
        return iter((self.status, self.report, self.state))


def load_objective(source: str | Path | dict | AOIInput) -> AOIInput:
    """Load and validate an objective from a file path, JSON string, dict, or AOIInput."""
    if isinstance(source, AOIInput):
        return source

    data: dict
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        # Check if source is a file path on disk
        if source_path.exists() and source_path.is_file():
            try:
                raw_text = source_path.read_text(encoding="utf-8")
            except Exception as exc:
                raise ObjectiveValidationError(
                    f"Could not read objective file {source}: {exc}"
                ) from exc
            try:
                data = json.loads(raw_text)
            except Exception as exc:
                raise ObjectiveValidationError(f"Invalid JSON syntax in {source}: {exc}") from exc
        else:
            # Check if source itself is a raw JSON string
            if isinstance(source, str) and source.strip().startswith("{"):
                try:
                    data = json.loads(source)
                except Exception as exc:
                    raise ObjectiveValidationError(f"Invalid JSON string: {exc}") from exc
            else:
                raise FileNotFoundError(f"Objective file not found: {source}")
    elif isinstance(source, dict):
        data = source
    else:
        raise ObjectiveValidationError(f"Unsupported objective source type: {type(source)}")

    if not isinstance(data, dict):
        raise ObjectiveValidationError("Objective data must be a JSON object.")

    # Support both full AOIInput format (with top-level 'objective' key)
    # and direct BusinessObjective format (where 'description' is top-level)
    try:
        if "objective" in data:
            return AOIInput.model_validate(data)
        elif "description" in data:
            obj = BusinessObjective.model_validate(data)
            return AOIInput(objective=obj)
        else:
            return AOIInput.model_validate(data)
    except ValidationError as err:
        raise ObjectiveValidationError(f"Objective validation failed: {err}") from err


class AOIRunner:
    """Thin run-execution layer for AOI.

    Orchestrates the lifecycle, invokes the compiled LangGraph pipeline,
    persists run records, and serializes reports without duplicating agent logic.
    """

    def __init__(
        self,
        db_session_factory=None,
        persist_to_db: bool = True,
    ):
        self.db_session_factory = db_session_factory
        self.persist_to_db = persist_to_db

    def run(
        self,
        objective: str | Path | dict | AOIInput,
        *,
        output_path: str | Path | None = None,
        run_id: str | None = None,
        discovery_provider=None,
        initial_candidates: list[Candidate] | None = None,
        propagate_interrupt: bool = False,
    ) -> RunResult:
        """Execute a full end-to-end AOI pipeline run."""
        # 1. Validate and load objective
        aoi_input = load_objective(objective)

        # 2. Initialize lifecycle state
        run_id = run_id or f"AOI-{uuid4().hex[:12]}"
        state = create_initial_state(aoi_input)
        state.run_id = run_id
        state.status = RunStatus.CREATED
        if initial_candidates:
            state.candidates = list(initial_candidates)

        started_at = datetime.now(UTC)

        # Optionally configure discovery provider if supplied
        if discovery_provider is not None:
            set_discovery_provider(discovery_provider)

        # Initial DB record
        self._record_run(
            run_id=run_id,
            status=RunStatus.CREATED,
            objective=aoi_input.objective.model_dump(),
            statistics={},
            started_at=started_at,
        )

        final_status = RunStatus.CREATED
        final_report: AOIReport | None = None

        # 3. Invoke compiled LangGraph pipeline
        try:
            try:
                graph_output = aoi_graph.invoke(state)
            finally:
                if discovery_provider is not None:
                    set_discovery_provider(None)

            # Extract status and report from output
            if isinstance(graph_output, dict):
                final_status = graph_output.get("status", RunStatus.COMPLETED)
                final_report = graph_output.get("report")
                # Rehydrate state with graph output updates
                state = state.model_copy(update=graph_output)
            elif isinstance(graph_output, AOIState):
                final_status = graph_output.status
                final_report = graph_output.report
                state = graph_output
            else:
                final_status = RunStatus.COMPLETED

        except KeyboardInterrupt as exc:
            logger.warning("Pipeline run %s interrupted by user.", run_id)
            final_status = RunStatus.CANCELLED
            state.status = RunStatus.CANCELLED
            final_report = None
            self._record_run(
                run_id=run_id,
                status=RunStatus.CANCELLED,
                objective=aoi_input.objective.model_dump(),
                statistics={},
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )
            if propagate_interrupt:
                raise exc
            return RunResult(
                run_id=run_id,
                status=RunStatus.CANCELLED,
                report=None,
                state=state,
            )

        except Exception as exc:
            logger.error("Pipeline run %s failed with exception: %s", run_id, exc)
            final_status = RunStatus.FAILED
            state.status = RunStatus.FAILED
            state.errors = list(state.errors) + [str(exc)]
            final_report = None
            self._record_run(
                run_id=run_id,
                status=RunStatus.FAILED,
                objective=aoi_input.objective.model_dump(),
                statistics={},
                started_at=started_at,
                completed_at=datetime.now(UTC),
            )
            return RunResult(
                run_id=run_id,
                status=RunStatus.FAILED,
                report=None,
                state=state,
            )

        completed_at = datetime.now(UTC)

        # 4. Final DB persistence
        stats_dict = final_report.statistics.model_dump() if final_report else {}
        self._record_run(
            run_id=run_id,
            status=final_status,
            objective=aoi_input.objective.model_dump(),
            statistics=stats_dict,
            started_at=started_at,
            completed_at=completed_at,
        )

        # 5. Output file serialization
        saved_file_path: Path | None = None
        if output_path and final_report is not None:
            saved_file_path = Path(output_path)
            saved_file_path.parent.mkdir(parents=True, exist_ok=True)
            report_json = final_report.model_dump_json(indent=2)
            saved_file_path.write_text(report_json, encoding="utf-8")

        return RunResult(
            run_id=run_id,
            status=final_status,
            report=final_report,
            state=state,
            output_file=saved_file_path,
        )

    def _record_run(
        self,
        run_id: str,
        status: RunStatus | str,
        objective: dict,
        statistics: dict,
        started_at: datetime,
        completed_at: datetime | None = None,
    ) -> None:
        """Safely persist run record using DB layer without crashing execution."""
        if not self.persist_to_db:
            return
        status_str = status.value if hasattr(status, "value") else str(status)
        try:
            from .db.session import persist_run

            persist_run(
                run_id=run_id,
                status=status_str,
                objective=objective,
                statistics=statistics,
                started_at=started_at,
                completed_at=completed_at,
                session_factory=self.db_session_factory,
            )
        except Exception as exc:
            logger.warning("Could not persist run %s to database: %s", run_id, exc)
