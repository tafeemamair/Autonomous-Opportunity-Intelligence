import logging
import os
import re
import time
from datetime import UTC, datetime
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from pydantic import HttpUrl

from ..config import get_settings
from ..schemas.common import SourceType
from ..schemas.discovery import Candidate, DiscoveryPlan
from .discovery import DiscoveryProvider

logger = logging.getLogger(__name__)

# Reference, government, encyclopedic, and generic regulatory domains to ignore as candidates
INFO_DOMAINS = {
    "wikipedia.org",
    "wikimedia.org",
    "wikidata.org",
    "federalregister.gov",
    "nist.gov",
    "usajobs.gov",
    "sec.gov",
    "whitehouse.gov",
    "congress.gov",
    "war.gov",
    "defense.gov",
    "ca.gov",
    "gov.uk",
    "archive.org",
}

# Major job boards, ATS, and aggregator domains where the publisher is NOT the candidate
JOB_BOARD_DOMAINS = {
    "ziprecruiter.com",
    "indeed.com",
    "dice.com",
    "linkedin.com",
    "getonbrd.com",
    "jobright.ai",
    "agentic-engineering-jobs.com",
    "glassdoor.com",
    "wellfound.com",
    "angel.co",
    "monster.com",
    "careerbuilder.com",
    "simplyhired.com",
    "builtin.com",
    "levels.fyi",
    "themuse.com",
    "hired.com",
}

# Directory, VC, and startup aggregator platforms
DIRECTORY_OR_AGGREGATOR_DOMAINS = {
    "vcbacked.co",
    "topstartups.io",
    "superscout.co",
    "aifundingtracker.com",
    "clutch.co",
    "g2.com",
    "capterra.com",
    "goodfirms.co",
    "crunchbase.com",
    "pitchbook.com",
    "ycombinator.com",
}

# News, media, and tech publications
NEWS_AND_MEDIA_DOMAINS = {
    "techcrunch.com",
    "venturebeat.com",
    "forbes.com",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "nytimes.com",
    "theverge.com",
    "wired.com",
    "businessinsider.com",
    "cnbc.com",
    "fortune.com",
    "zdnet.com",
    "infoworld.com",
    "arstechnica.com",
    "medium.com",
    "substack.com",
}

# Patterns indicating listicles, index pages, or generic roundups
LISTICLE_PATTERNS = [
    r"\b\d+\s+(?:best|top|greatest|promising)\b",
    r"\btop\s+\d+\b",
    r"\bbest\s+\d+\b",
    r"\b\d+\s+agentic\b",
    r"\b\d+\s+startups\b",
    r"\b\d+\s+companies\b",
    r"\blist\s+of\b",
    r"\bcompanies\s+buying\b",
    r"\bnow\s+hiring\b",
    r"\bjobs\s*\(\d+\b",
    r"\bbrowse\s+\d+\b",
    r"\bjobs\s+in\s+",
]

# Words and phrases that cannot be valid candidate company names
FORBIDDEN_COMPANY_NAMES = {
    "ziprecruiter",
    "indeed",
    "dice",
    "linkedin",
    "jobright",
    "usajobs",
    "getonbrd",
    "wikipedia",
    "nist",
    "federal register",
    "glassdoor",
    "wellfound",
    "y combinator",
    "techcrunch",
    "venturebeat",
    "forbes",
    "bloomberg",
    "reuters",
    "remote",
    "united states",
    "north america",
    "san francisco",
    "new york",
    "full time",
    "part time",
    "contract",
    "jobs",
    "careers",
    "hiring",
    "apply now",
    "now hiring",
    "overview",
    "salary",
    "work",
    "engineering",
    "ai agent",
    "ai agents",
    "artificial intelligence",
    "software development",
}


class TavilyProviderError(Exception):
    """Base exception for Tavily provider failures."""


class TavilyConfigurationError(TavilyProviderError):
    """Raised when Tavily configuration or API key is missing."""


class TavilyAuthenticationError(TavilyProviderError):
    """Raised when Tavily rejects the API key or authentication fails."""


class TavilyTransientError(TavilyProviderError):
    """Raised for transient issues like rate limits (429) or server errors (5xx)."""


class TavilyDiscoveryProvider(DiscoveryProvider):
    """Tavily search implementation of DiscoveryProvider.

    Converts search query results from the Tavily API into AOI Candidate entities,
    differentiating target prospect companies from publishers/aggregators,
    and preserving source URLs, metadata, and the originating discovery strategy.
    """

    DEFAULT_BASE_URL = "https://api.tavily.com"

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
        resolved_key = (
            api_key
            or settings.tavily_api_key
            or os.getenv("TAVILY_API_KEY")
            or os.getenv("AOI_TAVILY_API_KEY")
        )
        self.api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self.max_results_per_query = max_results_per_query
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self._custom_client = client
        self._internal_client: httpx.Client | None = None

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

    def __enter__(self) -> "TavilyDiscoveryProvider":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def discover(self, plan: DiscoveryPlan) -> list[Candidate]:
        """Execute discovery queries for each strategy in the plan and return normalized Candidates."""
        if not self.api_key:
            raise TavilyConfigurationError(
                "Tavily API key is missing. Set TAVILY_API_KEY in the environment or .env file."
            )

        candidates_by_key: dict[str, Candidate] = {}

        for strategy in plan.strategies:
            for query in strategy.query_templates:
                try:
                    search_response = self._search_with_retry(query)
                except TavilyAuthenticationError:
                    raise
                except Exception as exc:
                    logger.warning("Failed search for query '%s': %s", query, exc)
                    continue

                raw_results = search_response.get("results", [])
                for result in raw_results:
                    url = result.get("url")
                    title = result.get("title", "")
                    content = result.get("content", "")
                    if not url:
                        continue

                    self._process_result(
                        result_url=url,
                        title=title,
                        content=content,
                        query=query,
                        strategy_name=strategy.name,
                        candidates_by_key=candidates_by_key,
                    )

        return list(candidates_by_key.values())

    def _search_with_retry(self, query: str) -> dict:
        """Execute a Tavily search query with exponential backoff on transient errors."""
        endpoint = f"{self.base_url}/search"
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": self.max_results_per_query,
            "include_answer": False,
        }

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.post(endpoint, json=payload)
                if response.status_code == 200:
                    return response.json()

                if response.status_code in (401, 403):
                    raise TavilyAuthenticationError(
                        f"Tavily authentication failed ({response.status_code}): {response.text}"
                    )

                if response.status_code == 429 or response.status_code >= 500:
                    last_error = TavilyTransientError(
                        f"Tavily transient HTTP {response.status_code}: {response.text}"
                    )
                else:
                    raise TavilyProviderError(
                        f"Tavily query failed ({response.status_code}): {response.text}"
                    )

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_error = TavilyTransientError(f"Tavily network error: {exc}")

            if attempt < self.max_retries:
                sleep_time = self.backoff_factor * (2**attempt)
                time.sleep(sleep_time)

        raise last_error or TavilyProviderError("Tavily request failed after retries")

    def _process_result(
        self,
        result_url: str,
        title: str,
        content: str,
        query: str,
        strategy_name: str,
        candidates_by_key: dict[str, Candidate],
    ) -> None:
        """Parse search result into a candidate, ignoring non-prospect publishers and merging duplicates."""
        try:
            validated_source_url = HttpUrl(result_url)
        except Exception:
            return

        extracted = self._extract_candidate(
            result_url=result_url,
            title=title,
            content=content,
            strategy_name=strategy_name,
        )

        # If entity extraction is uncertain or points to a pure information/publisher site, skip it
        if extracted is None:
            return

        company_name, website, source_type = extracted

        # Deduplicate candidates: use website host if known, otherwise normalized company name
        if website is not None:
            dedup_key = website.host.lower()
        else:
            dedup_key = re.sub(r"[^a-z0-9]", "", company_name.lower())

        snippet = content[:200].strip() if content else ""
        reason = f"Discovered via {strategy_name} query '{query}': {title}."
        if snippet:
            reason = f"{reason} Snippet: {snippet}"

        if dedup_key in candidates_by_key:
            existing = candidates_by_key[dedup_key]
            if validated_source_url not in existing.source_urls:
                existing.source_urls.append(validated_source_url)
                existing.source_types.append(source_type)
            if existing.website is None and website is not None:
                existing.website = website
        else:
            candidate = Candidate(
                candidate_id=f"cand_{uuid4().hex[:12]}",
                company_name=company_name,
                website=website,
                location=None,
                discovery_strategy=strategy_name,
                discovery_reason=reason,
                source_urls=[validated_source_url],
                source_types=[source_type],
                discovered_at=datetime.now(UTC),
            )
            candidates_by_key[dedup_key] = candidate

    def _extract_candidate(
        self,
        result_url: str,
        title: str,
        content: str,
        strategy_name: str,
    ) -> tuple[str, HttpUrl | None, SourceType] | None:
        """Extract a distinct candidate target company and its website, or return None if uncertain."""
        parsed = urlparse(result_url)
        domain = parsed.netloc.lower()
        if not domain:
            return None
        if domain.startswith("www."):
            domain = domain[4:]

        path = parsed.path.lower()

        # 1. Skip government, regulatory, military, academic, and encyclopedia domains
        if (
            domain.endswith(".gov")
            or domain.endswith(".mil")
            or domain.endswith(".edu")
            or any(info_d in domain for info_d in INFO_DOMAINS)
        ):
            return None

        # 2. Check source categories
        is_job_board = any(jb in domain for jb in JOB_BOARD_DOMAINS) or any(
            ats in domain
            for ats in (
                "greenhouse.io",
                "lever.co",
                "ashbyhq.com",
                "smartrecruiters.com",
                "workable.com",
            )
        )

        is_news = (
            any(nm in domain for nm in NEWS_AND_MEDIA_DOMAINS)
            or "/news/" in path
            or "/press-releases/" in path
            or "/article/" in path
        )

        is_directory = any(d in domain for d in DIRECTORY_OR_AGGREGATOR_DOMAINS) or any(
            term in path for term in ("/directory/", "/sector/")
        )

        is_listicle = any(
            re.search(pat, title, re.IGNORECASE) for pat in LISTICLE_PATTERNS
        ) or any(term in path for term in ("/top-", "/best-", "/list-"))

        # 3. Handle Job Boards and ATS Platforms
        if is_job_board:
            # ATS platforms: boards.greenhouse.io/<company_slug>/... or jobs.lever.co/<company_slug>/...
            for ats in ("greenhouse.io", "lever.co", "ashbyhq.com", "workable.com"):
                if ats in domain:
                    slug_parts = [p for p in parsed.path.strip("/").split("/") if p]
                    if slug_parts and slug_parts[0] not in ("jobs", "careers", "search"):
                        name = slug_parts[0].replace("-", " ").replace("_", " ").title()
                        return name, None, SourceType.OFFICIAL_JOB

            # Pattern A: "<Role> at <Company>" or "<Role> @ <Company>"
            m_at = re.search(
                r"(?:hiring|engineer|developer|architect|lead|specialist|manager|consultant)\s+(?:at|@)\s+([A-Z0-9][A-Za-z0-9&.' -]{1,40})",
                title,
                re.IGNORECASE,
            )
            if m_at:
                cand = m_at.group(1).strip()
                if self._is_valid_extracted_company(cand):
                    return cand, None, SourceType.JOB_BOARD

            # Pattern B: "<Job Title> - <Company> - <Location>" (standard job boards)
            title_parts = re.split(r"\s+[-|—]\s+", title)
            if len(title_parts) >= 2:
                for candidate_part in title_parts[1:]:
                    cand = candidate_part.strip()
                    if self._is_valid_extracted_company(cand) and not self._is_likely_location(
                        cand
                    ):
                        return cand, None, SourceType.JOB_BOARD

            # Pattern C: Snippet metadata ("Company: <Company>" or "Employer: <Company>")
            m_snip = re.search(
                r"(?:Company|Employer|Organization):\s*([A-Z0-9][A-Za-z0-9&.' -]{1,40})",
                content,
                re.IGNORECASE,
            )
            if m_snip:
                cand = m_snip.group(1).strip()
                if self._is_valid_extracted_company(cand):
                    return cand, None, SourceType.JOB_BOARD

            # Pattern D: Markdown structured job snippets ("## <Title>\n\n<Company>\n\n/")
            m_md = re.search(
                r"##\s+[^\n]+\n+([A-Z0-9][A-Za-z0-9&.' -]{1,40})\n+/", content
            )
            if m_md:
                cand = m_md.group(1).strip()
                if self._is_valid_extracted_company(cand):
                    return cand, None, SourceType.JOB_BOARD

            # If no target company could be reliably extracted from a job board/aggregator, skip it
            return None

        # 4. Handle News and Media
        if is_news:
            # Headline subject actions: "<Company> raises...", "<Company> launches...", etc.
            m_news = re.search(
                r"^([A-Z0-9][A-Za-z0-9&.' -]{1,35}?)\s+(?:raises|launches|unveils|announces|secures|partners with|acquires|expands|deploys|integrates)\b",
                title.strip(),
                re.IGNORECASE,
            )
            if m_news:
                cand = m_news.group(1).strip()
                if self._is_valid_extracted_company(cand):
                    return cand, None, SourceType.NEWS

            # Headline intention: "<Company> to invest/build/launch..."
            m_to = re.search(
                r"^([A-Z0-9][A-Za-z0-9&.' -]{1,35}?)\s+to\s+(?:invest|build|launch|deploy|acquire)\b",
                title.strip(),
                re.IGNORECASE,
            )
            if m_to:
                cand = m_to.group(1).strip()
                if self._is_valid_extracted_company(cand):
                    return cand, None, SourceType.NEWS

            # General articles without a specific commercial actor should be skipped
            return None

        # 5. Handle Directories, Aggregators, and Listicles
        if is_directory or is_listicle:
            # Check for single-company profile pages on directories (e.g. ycombinator.com/companies/<slug>)
            if "ycombinator.com" in domain and "/companies/" in path:
                slug_parts = [p for p in path.split("/companies/")[1].split("/") if p]
                if slug_parts and slug_parts[0] not in (
                    "industry",
                    "location",
                    "all",
                    "search",
                    "founders",
                ):
                    name = slug_parts[0].replace("-", " ").title()
                    if self._is_valid_extracted_company(name):
                        return name, None, SourceType.DIRECTORY

            # Multi-company listicles or directory index pages are skipped
            return None

        # 6. Direct Company Websites
        company_name = self._extract_company_name(title=title, domain=domain)
        if not self._is_valid_extracted_company(company_name):
            return None

        try:
            website_url = HttpUrl(f"https://{domain}/")
        except Exception:
            website_url = None

        source_type = self._infer_source_type(result_url, strategy_name)
        return company_name, website_url, source_type

    @staticmethod
    def _is_valid_extracted_company(name: str) -> bool:
        """Validate whether an extracted string qualifies as a reasonable company name."""
        if not name or len(name) < 2 or len(name) > 40:
            return False
        clean = name.strip().lower()
        if clean in FORBIDDEN_COMPANY_NAMES:
            return False
        if any(
            term in clean
            for term in (
                "top ai",
                "best ai",
                "now hiring",
                "$",
                "/hr",
                "jobs in",
                "salaries",
                "reviews",
                "browse 1000+",
                "hiring jobs",
                "development company",
                "development companies",
            )
        ):
            return False
        return True

    @staticmethod
    def _is_likely_location(val: str) -> bool:
        """Check if a string looks like a location rather than a company name."""
        val_lower = val.lower().strip()
        return (
            any(
                loc in val_lower
                for loc in (
                    "remote",
                    "united states",
                    "usa",
                    "san francisco",
                    "new york",
                    "ca",
                    "ny",
                    "tx",
                    "fl",
                    "wa",
                )
            )
            and len(val.split()) <= 3
        )

    @staticmethod
    def _extract_company_name(title: str, domain: str) -> str:
        """Infer a readable company name from the page title or domain."""
        if title:
            # Common title separators: " | ", " - ", " : ", " — "
            parts = re.split(r"\s+[-|:—]\s+", title.strip())
            for part in parts:
                clean = part.strip()
                if not clean:
                    continue
                # Skip generic website noise words
                if (
                    re.search(
                        r"\b(home|careers|jobs|about|official|overview|hiring|login|platform|company)\b",
                        clean,
                        re.IGNORECASE,
                    )
                    and len(clean.split()) > 3
                ):
                    continue
                if 2 <= len(clean) <= 40:
                    return clean

        # Fallback to domain name without TLD
        domain_name = domain.split(".")[0]
        return domain_name.capitalize()

    @staticmethod
    def _infer_source_type(url: str, strategy_name: str) -> SourceType:
        """Infer SourceType based on URL and discovery strategy."""
        url_lower = url.lower()
        if any(
            term in url_lower
            for term in ("job", "career", "greenhouse.io", "lever.co", "ashbyhq")
        ):
            return SourceType.JOB_BOARD
        if "linkedin.com" in url_lower:
            return SourceType.LINKEDIN
        if strategy_name == "hiring_signals":
            return SourceType.OFFICIAL_JOB
        if strategy_name == "product_signals":
            return SourceType.OFFICIAL_ANNOUNCEMENT
        return SourceType.COMPANY_WEBSITE
