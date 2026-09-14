import argparse
import sys

# Ensure UTF-8 output encoding for web snippets on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from .agents.discovery import DiscoveryAgent
from .providers.tavily import TavilyConfigurationError, TavilyDiscoveryProvider, TavilyProviderError
from .schemas.objective import AOIInput, BusinessObjective


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
        print("Set your key via: export TAVILY_API_KEY='tvly-...' or pass --api-key", file=sys.stderr)
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


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="aoi",
        description="Autonomous Opportunity Intelligence (AOI) CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Discovery subcommand
    discover_parser = subparsers.add_parser(
        "discover",
        help="Run discovery using Tavily provider",
    )
    discover_parser.add_argument(
        "--objective",
        "-o",
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
        "--api-key",
        type=str,
        default=None,
        help="Tavily API key (overrides TAVILY_API_KEY env var)",
    )
    discover_parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum search results per query (default: 5)",
    )

    args = parser.parse_args()

    if args.command == "discover" or args.command is None:
        if args.command is None:
            # If no command passed, check if user provided arguments or display help
            if len(sys.argv) == 1:
                parser.print_help()
                sys.exit(0)

        sys.exit(
            run_discovery(
                objective=args.objective,
                market=args.market,
                opportunities=args.opportunities,
                api_key=args.api_key,
                max_results_per_query=args.max_results,
            )
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
