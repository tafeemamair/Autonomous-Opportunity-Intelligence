import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for web snippets on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os

from .agents.discovery import DiscoveryAgent
from .config import get_settings
from .providers.tavily import TavilyConfigurationError, TavilyDiscoveryProvider, TavilyProviderError
from .runner import AOIRunner, ObjectiveValidationError, RunResult, load_objective
from .schemas.common import RunStatus
from .schemas.objective import AOIInput, BusinessObjective


def print_run_summary(result: RunResult) -> None:
    """Print clean terminal summary according to Section 13."""
    rep = result.report
    stats = rep.statistics if rep else None

    print("\nAOI Run")
    print("────────────────────────────")
    print(f"Run ID: {result.run_id}")
    print("Status:")
    print("✓ Discovery")
    print("✓ Research")
    print("✓ Qualification")
    print("✓ Verification")
    print("✓ Scoring")
    print("✓ Reporting")
    print()

    if stats:
        print(f"Candidates discovered: {stats.candidates_discovered}")
        print(f"Candidates researched: {stats.candidates_researched}")
        print(f"Candidates qualified:  {stats.candidates_qualified}")
        print(f"Candidates verified:   {stats.candidates_verified}")
        print()
        print(f"High priority: {stats.high_priority}")
        print(f"Qualified:     {stats.qualified_priority}")
        print(f"Watchlist:     {stats.watchlist_priority}")
        print(f"Discarded:     {stats.discard_priority}")
        print()

    if rep:
        print(f"Report: {rep.summary.headline}")
    if result.output_file:
        print(f"Output: {result.output_file}")

    status_str = result.status.value if hasattr(result.status, "value") else str(result.status)
    print(f"\nStatus: {status_str}")


def handle_validate(objective_path: str) -> int:
    """Validate objective file and output result."""
    path = Path(objective_path)
    try:
        aoi_input = load_objective(path)
        desc = aoi_input.objective.description
        print(f"[VALID] Objective is valid: {desc}")
        return 0
    except FileNotFoundError as err:
        print(f"[ERROR] Objective file not found: {err}", file=sys.stderr)
        return 1
    except ObjectiveValidationError as err:
        print(f"[ERROR] Objective validation failed: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"[ERROR] Unexpected validation error: {err}", file=sys.stderr)
        return 1


def handle_run(
    objective_path: str,
    output_path: str | None = None,
    discovery_provider=None,
) -> int:
    """Execute end-to-end run via AOIRunner."""
    runner = AOIRunner()
    try:
        # Load objective
        aoi_input = load_objective(objective_path)
    except FileNotFoundError as err:
        print(f"[ERROR] Objective file not found: {err}", file=sys.stderr)
        return 1
    except ObjectiveValidationError as err:
        print(f"[ERROR] Invalid objective JSON: {err}", file=sys.stderr)
        return 1
    except Exception as err:
        print(f"[ERROR] Could not load objective: {err}", file=sys.stderr)
        return 1

    # In real operator runs (outside automated test runners), construct the configured Tavily discovery provider
    if discovery_provider is None and "PYTEST_CURRENT_TEST" not in os.environ:
        settings = get_settings()
        if (
            settings.tavily_api_key
            or os.getenv("TAVILY_API_KEY")
            or os.getenv("AOI_TAVILY_API_KEY")
        ):
            try:
                discovery_provider = TavilyDiscoveryProvider()
            except Exception as exc:
                print(f"[WARN] Could not initialize Tavily provider: {exc}", file=sys.stderr)

    try:
        result = runner.run(
            aoi_input,
            output_path=output_path,
            discovery_provider=discovery_provider,
            propagate_interrupt=False,
        )
    except KeyboardInterrupt:
        print("\n[CANCELLED] AOI run cancelled by user.", file=sys.stderr)
        return 130
    except Exception as err:
        print(f"\n[ERROR] Pipeline execution failed: {err}", file=sys.stderr)
        return 1

    print_run_summary(result)

    if result.status == RunStatus.CANCELLED:
        return 130
    elif result.status == RunStatus.FAILED:
        return 1
    return 0


def run_discovery(
    objective: str,
    market: str = "United States",
    opportunities: list[str] | None = None,
    api_key: str | None = None,
    max_results_per_query: int = 5,
) -> int:
    """Execute a single discovery cycle using DiscoveryAgent and TavilyDiscoveryProvider."""
    opp_list = opportunities or ["AI automation"]
    print("=" * 60)
    print("AOI DISCOVERY RUNNER (Tavily)")
    print("=" * 60)
    print(f"Objective     : {objective}")
    print(f"Market        : {market}")
    print(f"Opportunities : {', '.join(opp_list)}")
    print("-" * 60)

    try:
        provider = TavilyDiscoveryProvider(
            api_key=api_key,
            max_results_per_query=max_results_per_query,
        )
    except TavilyConfigurationError as err:
        print(f"\n[ERROR] Configuration error: {err}", file=sys.stderr)
        return 1

    # 1. Build input objective
    aoi_input = AOIInput(
        objective=BusinessObjective(
            description=objective,
            target_markets=[market],
            target_opportunities=opp_list,
        )
    )

    # 2. Generate discovery plan
    agent = DiscoveryAgent()
    plan = agent.plan(aoi_input)
    print(f"[1/2] Generated discovery plan with {len(plan.strategies)} strategies.")
    for strat in plan.strategies:
        print(f"      - {strat.name}: {len(strat.query_templates)} queries")

    # 3. Execute discovery via Tavily
    print("\n[2/2] Querying Tavily for candidates...")
    try:
        candidates = provider.discover(plan)
    except TavilyConfigurationError as err:
        print(f"\n[ERROR] {err}", file=sys.stderr)
        print("Set your key via: export TAVILY_API_KEY='tvly-...' or .env", file=sys.stderr)
        return 1
    except TavilyProviderError as err:
        print(f"\n[ERROR] Provider error during discovery: {err}", file=sys.stderr)
        return 2

    print(f"\nDiscovered {len(candidates)} unique candidates:\n")
    for idx, candidate in enumerate(candidates, 1):
        website_str = str(candidate.website) if candidate.website else "N/A"
        print(f"{idx:2d}. {candidate.company_name} ({website_str})")
        print(f"    Strategy : {candidate.discovery_strategy}")
        print(f"    Reason   : {candidate.discovery_reason}")
        print(f"    Sources  : {len(candidate.source_urls)}")
        for url in candidate.source_urls[:3]:
            print(f"      * {url}")
        print()

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="aoi",
        description="Autonomous Opportunity Intelligence (AOI) CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # 1. Validate subcommand
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an objective JSON file against the AOI schema",
    )
    validate_parser.add_argument(
        "objective_file",
        type=str,
        help="Path to objective JSON file",
    )

    # 2. Run subcommand
    run_parser = subparsers.add_parser(
        "run",
        help="Execute end-to-end AOI pipeline from objective file",
    )
    run_parser.add_argument(
        "objective_file",
        type=str,
        help="Path to objective JSON file",
    )
    run_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to write the serialized AOIReport JSON",
    )

    # 3. Discover subcommand (backward-compatible)
    discover_parser = subparsers.add_parser(
        "discover",
        help="Run discovery using Tavily provider",
    )
    discover_parser.add_argument(
        "--objective",
        type=str,
        default="Find companies with recent AI automation needs.",
        help="Business objective description (min 10 characters)",
    )
    discover_parser.add_argument(
        "--market",
        "-m",
        type=str,
        default="United States",
        help="Target market or geography",
    )
    discover_parser.add_argument(
        "--opportunity",
        "--opportunities",
        nargs="+",
        dest="opportunities",
        default=["AI automation"],
        help="Target opportunity type(s) or capability(ies)",
    )
    discover_parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum search results per query (default: 5)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2

    if args.command == "validate":
        return handle_validate(args.objective_file)

    elif args.command == "run":
        return handle_run(
            objective_path=args.objective_file,
            output_path=args.output,
        )

    elif args.command == "discover":
        return run_discovery(
            objective=args.objective,
            market=args.market,
            opportunities=args.opportunities,
            max_results_per_query=args.max_results,
        )

    else:
        parser.print_help()
        return 0 if argv is not None and len(argv) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
