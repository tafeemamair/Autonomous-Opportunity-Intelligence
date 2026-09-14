import logging
import re
from datetime import UTC, datetime
from urllib.parse import urlparse

from pydantic import HttpUrl

from ..providers.research import ResearchItem, ResearchProvider
from ..schemas.common import Evidence, Source, SourceType, VerificationStatus
from ..schemas.discovery import Candidate
from ..schemas.opportunity import DecisionMaker
from ..schemas.research import ResearchResult, ResearchStatus

logger = logging.getLogger(__name__)

OFFICIAL_DOMAINS_EXCLUDE = {
    "wikipedia.org",
    "github.com",
    "linkedin.com",
    "twitter.com",
    "x.com",
    "facebook.com",
    "youtube.com",
    "instagram.com",
    "glassdoor.com",
    "indeed.com",
    "ziprecruiter.com",
    "jobright.ai",
    "techcrunch.com",
    "forbes.com",
    "bloomberg.com",
    "clutch.co",
    "leadiq.com",
    "g2.com",
    "zoominfo.com",
    "crunchbase.com",
    "pitchbook.com",
    "prnewswire.com",
    "businesswire.com",
}

ROLE_PREFIXES = {
    "Senior",
    "Sr",
    "Sr.",
    "Executive",
    "Exec",
    "Exec.",
    "Chief",
    "Principal",
    "Associate",
    "Assoc",
    "Global",
    "Group",
    "Lead",
}

STOCK_IMAGE_PATTERNS = [
    r"\b(?:young|male|female|smiling|serious)?\s*(?:man|woman|engineer|developer|person|worker|businessman|businesswoman|professional)\s+(?:working on|sitting at|looking at|posing in|standing in|holding|typing on|using)\b",
    r"\b(?:working on computer|working on laptop|sitting in front of computer)\b",
    r"\b(?:close-up of|photo of|portrait of|view of|image of|picture of|stock photo|illustration of|rear view of|side view of)\b",
    r"\b(?:in a modern office|in a technological office|in an office setting)\b",
    r"\b(?:holding tablet|looking at screen|looking at monitor|wearing headset)\b",
]


class ResearchAgent:
    """Investigates individual Candidate entities and extracts source-backed Evidence.

    Does NOT decide qualification, scoring, or priority. Gathers factual signals
    around company identity, business events, automation needs, and decision makers.
    """

    def __init__(
        self,
        provider: ResearchProvider | None = None,
        max_queries: int = 4,
        recency_days: int = 90,
    ):
        self._provider = provider
        self.max_queries = max_queries
        self.recency_days = recency_days

    @property
    def provider(self) -> ResearchProvider:
        if self._provider is None:
            # Lazy import to avoid circular dependency
            from ..providers.tavily import TavilyResearchProvider

            self._provider = TavilyResearchProvider()
        return self._provider

    def generate_research_queries(self, candidate: Candidate) -> list[str]:
        """Generate deterministic, focused research queries for a candidate."""
        name = candidate.company_name
        return [
            f'"{name}" official website',
            f'"{name}" "AI"',
            f'"{name}" "AI agents"',
            f'"{name}" automation',
            f'"{name}" hiring jobs',
            f'"{name}" engineering',
            f'"{name}" announcement launch funding',
        ]

    def research(self, candidate: Candidate) -> ResearchResult:
        """Execute research on one candidate, extract structured evidence, and return ResearchResult."""
        candidate_id = candidate.candidate_id
        company_name = candidate.company_name
        website = candidate.website
        location = candidate.location
        industry: str | None = None
        people: list[DecisionMaker] = []
        technology_signals: list[str] = []
        business_signals: list[str] = []
        problem_signals: list[str] = []
        evidence_list: list[Evidence] = []
        warnings: list[str] = []

        # Track official domain for accurate source classification
        official_domain = (
            urlparse(str(website)).netloc.lower().removeprefix("www.")
            if website
            else None
        )

        # 1. Preserve initial candidate discovery signals as first evidence item if present
        if candidate.source_urls and candidate.discovery_reason:
            primary_url = candidate.source_urls[0]
            src_type = (
                candidate.source_types[0]
                if candidate.source_types
                else SourceType.OTHER
            )
            disc_source = Source(
                url=primary_url,
                source_type=src_type,
                title=f"Discovery Source: {candidate.discovery_strategy}",
                publisher=urlparse(str(primary_url)).netloc,
                published_at=None,
                accessed_at=candidate.discovered_at,
                reliability=self._compute_reliability(src_type),
            )
            disc_claim = f"{company_name} discovered via {candidate.discovery_strategy}: {candidate.discovery_reason}"
            evidence_list.append(
                Evidence(
                    claim=disc_claim,
                    source=disc_source,
                    evidence_summary=candidate.discovery_reason[:300],
                    verification_status=VerificationStatus.UNVERIFIED,
                    recency_score=85,
                )
            )

        # 2. Generate and execute search queries
        all_queries = self.generate_research_queries(candidate)
        selected_queries = all_queries[: self.max_queries]
        retrieved_items: list[tuple[str, ResearchItem]] = []

        for query in selected_queries:
            try:
                items = self.provider.search(query, max_results=3)
                for item in items:
                    retrieved_items.append((query, item))
            except Exception as exc:
                msg = f"Search query '{query}' failed: {exc}"
                logger.warning(msg)
                warnings.append(msg)

        # 3. Analyze retrieved items and extract structured facts
        seen_urls: set[str] = set()

        for query, item in retrieved_items:
            item_url_str = str(item.url)
            if item_url_str in seen_urls:
                continue
            seen_urls.add(item_url_str)

            parsed = urlparse(item_url_str)
            domain = parsed.netloc.lower().removeprefix("www.")
            content = item.content or ""
            title = item.title or ""
            published_at = self._parse_published_date(item.published_date)

            source_type = self._infer_source_type(
                item_url_str,
                domain,
                company_name=company_name,
                official_domain=official_domain,
            )
            reliability = self._compute_reliability(source_type)
            recency = self._compute_recency_score(published_at)

            # A. Attempt to identify official website if not already known
            if website is None and self._looks_like_official_domain(domain, company_name):
                try:
                    official_url = HttpUrl(f"https://{domain}/")
                    website = official_url
                    official_domain = domain
                    claim = f"{company_name} official website identified at {official_url}"
                    source = Source(
                        url=item.url,
                        source_type=SourceType.COMPANY_WEBSITE,
                        title=title,
                        publisher=domain,
                        published_at=published_at,
                        accessed_at=datetime.now(UTC),
                        reliability=95,
                    )
                    evidence_list.append(
                        Evidence(
                            claim=claim,
                            source=source,
                            evidence_summary=f"Official company website: {official_url}",
                            verification_status=VerificationStatus.UNVERIFIED,
                            recency_score=100,
                        )
                    )
                except Exception:
                    pass

            # B. Extract Location if not known
            if location is None:
                loc_match = re.search(
                    r"(?:headquartered in|based in|located in|headquarters in)\s+([A-Z][A-Za-z0-9\s,]+?)(?=[.;\n]|,\s*(?:the\s+company|a\s+company|which|where|with|operating|employing|founded|and\b)|(?:\s+(?:and|with|founded|co-founded|operating|building|offering))\b|$)",
                    content,
                    re.IGNORECASE,
                )
                if loc_match:
                    found_loc = self._clean_location(loc_match.group(1))
                    if len(found_loc) > 2:
                        location = found_loc
                        claim = f"{company_name} is located in {found_loc}."
                        source = Source(
                            url=item.url,
                            source_type=source_type,
                            title=title,
                            publisher=domain,
                            published_at=published_at,
                            accessed_at=datetime.now(UTC),
                            reliability=reliability,
                        )
                        evidence_list.append(
                            Evidence(
                                claim=claim,
                                source=source,
                                evidence_summary=f"Headquarters/location evidence: {found_loc}",
                                verification_status=VerificationStatus.UNVERIFIED,
                                recency_score=recency,
                            )
                        )

            # C. Extract People / Decision Makers
            people_found = self._extract_people(content, company_name, item.url)
            for person in people_found:
                if not any(p.name == person.name for p in people):
                    people.append(person)
                    claim = f"{person.name} is identified as {person.role} at {company_name}."
                    source = Source(
                        url=item.url,
                        source_type=source_type,
                        title=title,
                        publisher=domain,
                        published_at=published_at,
                        accessed_at=datetime.now(UTC),
                        reliability=reliability,
                    )
                    evidence_list.append(
                        Evidence(
                            claim=claim,
                            source=source,
                            evidence_summary=f"Leadership role: {person.name} ({person.role})",
                            verification_status=VerificationStatus.UNVERIFIED,
                            recency_score=recency,
                        )
                    )

            # D. Extract Technology / AI Agent Signals
            tech_match = re.search(
                r"([^.?!;\n]*\b(?:AI agent|AI agents|agentic|autonomous agent|LLM|GenAI)\b[^.?!;\n]*)",
                content,
                re.IGNORECASE,
            )
            if tech_match:
                signal_text = tech_match.group(1).strip()
                if (
                    len(signal_text) > 15
                    and not self._is_stock_image_or_caption(signal_text)
                    and signal_text not in technology_signals
                ):
                    technology_signals.append(signal_text)
                    problem_signals.append(f"AI agent capability/need: {signal_text}")
                    claim = f"{company_name} has active AI/agent development signals: '{signal_text}'."
                    source = Source(
                        url=item.url,
                        source_type=source_type,
                        title=title,
                        publisher=domain,
                        published_at=published_at,
                        accessed_at=datetime.now(UTC),
                        reliability=reliability,
                    )
                    evidence_list.append(
                        Evidence(
                            claim=claim,
                            source=source,
                            evidence_summary=signal_text[:300],
                            verification_status=VerificationStatus.UNVERIFIED,
                            recency_score=recency,
                        )
                    )

            # E. Extract Automation & Problem Signals
            auto_match = re.search(
                r"([^.?!;\n]*\b(?:workflow automation|automate|repetitive tasks|operational bottlenecks|internal tooling)\b[^.?!;\n]*)",
                content,
                re.IGNORECASE,
            )
            if auto_match:
                auto_text = auto_match.group(1).strip()
                if (
                    len(auto_text) > 15
                    and not self._is_stock_image_or_caption(auto_text)
                    and auto_text not in problem_signals
                ):
                    problem_signals.append(auto_text)
                    claim = f"{company_name} has operational workflow automation needs/signals: '{auto_text}'."
                    source = Source(
                        url=item.url,
                        source_type=source_type,
                        title=title,
                        publisher=domain,
                        published_at=published_at,
                        accessed_at=datetime.now(UTC),
                        reliability=reliability,
                    )
                    evidence_list.append(
                        Evidence(
                            claim=claim,
                            source=source,
                            evidence_summary=auto_text[:300],
                            verification_status=VerificationStatus.UNVERIFIED,
                            recency_score=recency,
                        )
                    )

            # F. Extract Business & Hiring Signals
            hire_match = re.search(
                r"([^.?!;\n]*\b(?:hiring|open role|seeking|job opening|careers|engineer|developer)\b[^.?!;\n]*)",
                content,
                re.IGNORECASE,
            )
            if hire_match:
                hire_text = hire_match.group(1).strip()
                if (
                    len(hire_text) > 15
                    and not self._is_stock_image_or_caption(hire_text)
                    and hire_text not in business_signals
                ):
                    business_signals.append(hire_text)
                    claim = f"{company_name} has active engineering or hiring demand: '{hire_text}'."
                    source = Source(
                        url=item.url,
                        source_type=source_type,
                        title=title,
                        publisher=domain,
                        published_at=published_at,
                        accessed_at=datetime.now(UTC),
                        reliability=reliability,
                    )
                    evidence_list.append(
                        Evidence(
                            claim=claim,
                            source=source,
                            evidence_summary=hire_text[:300],
                            verification_status=VerificationStatus.UNVERIFIED,
                            recency_score=recency,
                        )
                    )

        # 4. Determine status based on search and evidence results
        if warnings and not retrieved_items and not evidence_list:
            status = ResearchStatus.FAILED
        elif warnings or len(evidence_list) < 2:
            status = ResearchStatus.PARTIAL
        else:
            status = ResearchStatus.COMPLETED

        return ResearchResult(
            candidate_id=candidate_id,
            company_name=company_name,
            website=website,
            location=location,
            industry=industry,
            people=people,
            technology_signals=technology_signals,
            business_signals=business_signals,
            problem_signals=problem_signals,
            evidence=evidence_list,
            warnings=warnings,
            research_status=status,
            researched_at=datetime.now(UTC),
        )

    def _extract_people(
        self, content: str, company_name: str, source_url: HttpUrl
    ) -> list[DecisionMaker]:
        """Extract high-confidence decision maker profiles without inventing data."""
        found: list[DecisionMaker] = []

        patterns = [
            (
                r"\b(?:founded by|co-founded by|co-founders?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
                "Founder",
            ),
            (
                r"\b(?:CEO|Chief Executive Officer)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
                "CEO",
            ),
            (
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}),?\s+(?:is the\s+)?(?:CEO|Chief Executive Officer)\b",
                "CEO",
            ),
            (
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}),?\s+(?:is the\s+)?(?:CTO|Chief Technology Officer)\b",
                "CTO",
            ),
            (
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}),?\s+(?:is the\s+)?(?:VP of Engineering|Vice President of Engineering)\b",
                "VP of Engineering",
            ),
            (
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}),?\s+(?:is the\s+)?(?:Head of AI|Director of AI)\b",
                "Head of AI",
            ),
        ]

        for pat, role in patterns:
            for match in re.finditer(pat, content, re.IGNORECASE):
                raw_name = match.group(1).strip()
                name = re.sub(r"\s+", " ", raw_name).strip(" ,.-")
                current_role = re.sub(r"\s+", " ", role).strip(" ,.-")

                # Handle role prefixes that attached to the name (e.g. "Toby Roberts\n\nSenior")
                words = name.split()
                prefixes: list[str] = []
                while len(words) >= 2 and words[-1] in ROLE_PREFIXES:
                    prefixes.insert(0, words.pop())
                if prefixes:
                    name = " ".join(words)
                    current_role = f"{' '.join(prefixes)} {current_role}"

                # Skip invalid names matching company itself or generic nouns
                if (
                    name.lower() in company_name.lower()
                    or company_name.lower() in name.lower()
                    or name.lower()
                    in ("artificial intelligence", "united states", "san francisco")
                ):
                    continue
                if 4 <= len(name) <= 35 and not any(p.name == name for p in found):
                    found.append(
                        DecisionMaker(
                            name=name, role=current_role, profile_url=None
                        )
                    )

        return found

    @staticmethod
    def _looks_like_official_domain(domain: str, company_name: str) -> bool:
        """Check if a search result domain likely represents the official company website."""
        if any(ex in domain for ex in OFFICIAL_DOMAINS_EXCLUDE):
            return False
        clean_company = re.sub(r"[^a-z0-9]", "", company_name.lower())
        domain_name = domain.split(".")[0]
        return clean_company in domain_name or domain_name in clean_company

    @staticmethod
    def _clean_location(raw_loc: str) -> str:
        """Sanitize location and strip trailing non-location clauses."""
        raw_loc = raw_loc.strip().rstrip(".,;")
        parts = [p.strip() for p in raw_loc.split(",") if p.strip()]
        if len(parts) <= 1:
            return raw_loc
        result_parts = [parts[0]]
        for p in parts[1:]:
            if re.match(
                r"^(?:the\s+company|a\s+company|which|where|with|operating|employing|founded|and|currently|spanning)\b",
                p,
                re.IGNORECASE,
            ):
                break
            if len(p) > 35 or re.search(r"\b(?:has|is|was|were|are|have)\b", p, re.IGNORECASE):
                break
            result_parts.append(p)
        return ", ".join(result_parts)

    @staticmethod
    def _is_stock_image_or_caption(text: str) -> bool:
        """Detect obvious stock image alt-text or caption noise."""
        return any(re.search(pat, text, re.IGNORECASE) for pat in STOCK_IMAGE_PATTERNS)

    @staticmethod
    def _infer_source_type(
        url: str,
        domain: str,
        company_name: str | None = None,
        official_domain: str | None = None,
    ) -> SourceType:
        """Assign SourceType adhering to AOI evidence tiers."""
        url_lower = url.lower()

        # Tier 5: Social / Video platforms
        if any(
            s_d in domain
            for s_d in (
                "youtube.com",
                "instagram.com",
                "twitter.com",
                "x.com",
                "facebook.com",
                "reddit.com",
                "tiktok.com",
            )
        ):
            return SourceType.SOCIAL

        # Tier 3: Professional Networks & Job Boards
        if "linkedin.com" in domain:
            return SourceType.LINKEDIN
        if any(
            term in url_lower
            for term in (
                "job",
                "career",
                "greenhouse.io",
                "lever.co",
                "ashbyhq",
                "indeed.com",
                "ziprecruiter.com",
                "jobright.ai",
                "dice.com",
            )
        ):
            return SourceType.JOB_BOARD

        # Tier 4: Directories & Aggregators
        if any(
            d_d in domain
            for d_d in (
                "clutch.co",
                "leadiq.com",
                "g2.com",
                "zoominfo.com",
                "crunchbase.com",
                "pitchbook.com",
                "apollo.io",
            )
        ):
            return SourceType.DIRECTORY

        # Tier 1: Official Announcements / Press Wires
        if (
            any(
                pr_d in domain
                for pr_d in (
                    "prnewswire.com",
                    "businesswire.com",
                    "globenewswire.com",
                    "accesswire.com",
                )
            )
            or "/press-releases/" in url_lower
            or "/announcements/" in url_lower
            or "/news-releases/" in url_lower
        ):
            return SourceType.OFFICIAL_ANNOUNCEMENT

        # Tier 2: Major News & Publications
        if any(
            news_d in domain
            for news_d in (
                "techcrunch.com",
                "forbes.com",
                "bloomberg.com",
                "reuters.com",
                "wsj.com",
                "nytimes.com",
                "cnbc.com",
            )
        ):
            return SourceType.NEWS
        if any(
            ind_d in domain
            for ind_d in (
                "venturebeat.com",
                "wired.com",
                "zdnet.com",
                "informationweek.com",
            )
        ):
            return SourceType.INDUSTRY_PUBLICATION

        # Tier 1: Candidate Official Company Website
        # Only classify if it actually matches the candidate's known official domain
        # or clearly matches the candidate's company name.
        if official_domain and (
            domain == official_domain or domain.endswith("." + official_domain)
        ):
            return SourceType.COMPANY_WEBSITE
        if company_name and ResearchAgent._looks_like_official_domain(
            domain, company_name
        ):
            return SourceType.COMPANY_WEBSITE

        # Unknown / third-party source
        return SourceType.OTHER

    def _compute_reliability(self, source_type: SourceType) -> int:
        """Map source type to explicit reliability score according to defined tiers."""
        match source_type:
            case SourceType.COMPANY_WEBSITE | SourceType.REGULATORY:
                return 95  # Tier 1
            case SourceType.OFFICIAL_JOB | SourceType.OFFICIAL_ANNOUNCEMENT:
                return 90  # Tier 1
            case SourceType.NEWS:
                return 80  # Tier 2
            case SourceType.INDUSTRY_PUBLICATION:
                return 75  # Tier 2
            case SourceType.LINKEDIN:
                return 65  # Tier 3
            case SourceType.JOB_BOARD:
                return 60  # Tier 3
            case SourceType.DIRECTORY:
                return 45  # Tier 4
            case SourceType.SOCIAL:
                return 25  # Tier 5
            case _:
                return 50

    def _compute_recency_score(self, published_at: datetime | None) -> int:
        """Calculate recency score prioritizing findings within recency_days (default 90)."""
        if not published_at:
            return 50
        days_ago = (datetime.now(UTC) - published_at).days
        if days_ago <= self.recency_days:
            return 95
        if days_ago <= 180:
            return 75
        if days_ago <= 365:
            return 50
        return 30

    @staticmethod
    def _parse_published_date(date_str: str | None) -> datetime | None:
        """Parse ISO / standard published date strings safely."""
        if not date_str:
            return None
        clean_date = date_str.strip()
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y/%m/%d"):
            try:
                dt = datetime.strptime(clean_date, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=UTC)
                return dt
            except ValueError:
                continue
        return None
