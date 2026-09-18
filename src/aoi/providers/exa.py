import logging
import os
import time

import httpx

from ..config import get_settings
from ..schemas.discovery import Candidate, DiscoveryBudget, DiscoveryPlan
from .discovery import DiscoveryProvider
from .tavily import TavilyDiscoveryProvider

logger = logging.getLogger(__name__)


class ExaProviderError(Exception):
    """Base exception for Exa provider failures."""


class ExaConfigurationError(ExaProviderError):
    """Raised when Exa configuration or API key is missing."""


class ExaAuthenticationError(ExaProviderError):
    """Raised when Exa rejects the API key."""


class ExaTransientError(ExaProviderError):
    """Raised for transient Exa errors such as rate limits or server errors."""


class ExaDiscoveryProvider(DiscoveryProvider):
    """Exa web-search adapter that reuses AOI's existing candidate normalization."""

    DEFAULT_BASE_URL = "https://api.exa.ai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        max_results_per_query: int = 5,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        timeout: float = 15.0,
        client: httpx.Client | None = None,
    ):
        settings = get_settings()
        self.api_key = (
            api_key
            or settings.exa_api_key
            or os.getenv("EXA_API_KEY")
            or os.getenv("AOI_EXA_API_KEY")
        )
        self.base_url = base_url.rstrip("/")
        self.max_results_per_query = max_results_per_query
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self._custom_client = client
        self._internal_client: httpx.Client | None = None
        # Reuse the proven AOI result-to-Candidate parsing rules until that parser
        # is deliberately extracted into a provider-neutral component.
        self._parser = TavilyDiscoveryProvider(api_key="parser-only")

    @property
    def client(self) -> httpx.Client:
        if self._custom_client is not None:
            return self._custom_client
        if self._internal_client is None:
            self._internal_client = httpx.Client(timeout=self.timeout)
        return self._internal_client

    def close(self) -> None:
        if self._internal_client is not None:
            self._internal_client.close()
            self._internal_client = None

    def __enter__(self) -> "ExaDiscoveryProvider":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def discover(
        self,
        plan: DiscoveryPlan,
        budget: DiscoveryBudget | None = None,
    ) -> list[Candidate]:
        if not self.api_key:
            raise ExaConfigurationError(
                "Exa API key is missing. Set EXA_API_KEY in the environment or .env file."
            )

        resolved_budget = budget or plan.budget or DiscoveryBudget(
            max_results_per_query=self.max_results_per_query,
        )
        candidates_by_key: dict[str, Candidate] = {}
        queries_executed = 0

        for strategy in plan.strategies:
            if queries_executed >= resolved_budget.max_queries:
                break

            for query in strategy.query_templates:
                if queries_executed >= resolved_budget.max_queries:
                    break

                queries_executed += 1
                try:
                    response = self._search_with_retry(
                        query, max_results=resolved_budget.max_results_per_query
                    )
                except ExaAuthenticationError:
                    raise
                except Exception as exc:
                    logger.warning("Failed Exa search for query '%s': %s", query, exc)
                    continue

                for result in response.get("results", []):
                    url = result.get("url")
                    if not url:
                        continue

                    content = result.get("text") or result.get("summary") or ""
                    if not content and result.get("highlights"):
                        content = " ".join(result["highlights"])

                    self._parser._process_result(
                        result_url=url,
                        title=result.get("title", ""),
                        content=content,
                        query=query,
                        strategy_name=strategy.name,
                        candidates_by_key=candidates_by_key,
                    )

        return list(candidates_by_key.values())[: resolved_budget.max_candidates]

    def _search_with_retry(self, query: str, max_results: int | None = None) -> dict:
        endpoint = f"{self.base_url}/search"
        payload = {
            "query": query,
            "type": "auto",
            "numResults": max_results or self.max_results_per_query,
            "contents": {
                "text": True,
                "highlights": True,
            },
        }
        headers = {
            "x-api-key": self.api_key or "",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(endpoint, json=payload, headers=headers)

                if response.status_code == 200:
                    return response.json()

                if response.status_code in (401, 402, 403):
                    raise ExaAuthenticationError(
                        f"Exa authentication failed ({response.status_code}): {response.text}"
                    )

                if response.status_code == 429 or response.status_code >= 500:
                    last_error = ExaTransientError(
                        f"Exa transient HTTP {response.status_code}: {response.text}"
                    )
                else:
                    raise ExaProviderError(
                        f"Exa query failed ({response.status_code}): {response.text}"
                    )

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = ExaTransientError(f"Exa network error: {exc}")

            if attempt < self.max_retries:
                time.sleep(self.backoff_factor * (2**attempt))

        raise last_error or ExaProviderError("Exa request failed after retries")
