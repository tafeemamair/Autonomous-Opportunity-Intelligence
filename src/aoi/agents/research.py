import logging
import re
from datetime import UTC, datetime
from urllib.parse import urlparse

from pydantic import HttpUrl

from ..providers.research import ResearchProvider
from ..schemas.common import Evidence, Source, SourceType, VerificationStatus
from ..schemas.discovery import Candidate
from ..schemas.opportunity import DecisionMaker
from ..schemas.research import (
    ResearchEvaluation,
    ResearchQuality,
    ResearchResult,
    ResearchStatus,
)

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

INDUSTRY_PATTERNS = [
    (r"\b(?:autonomous marine|subsea|ocean exploration|seafloor)\b", "Autonomous Marine Robotics"),
    (r"\b(?:robotics|hardware automation|drones|uav|autonomous vehicle)\b", "Robotics & Automation"),
    (r"\b(?:fintech|banking|payments|lending|wealth management)\b", "Fintech & Financial Services"),
    (r"\b(?:healthcare|biotech|medical|clinical|pharma|genomics)\b", "Healthcare & Life Sciences"),
    (r"\b(?:devtools|developer tools|software development|cloud infrastructure|observability)\b", "DevTools & Infrastructure"),
    (r"\b(?:ecommerce|e-commerce|retail|marketplace|direct-to-consumer)\b", "E-Commerce & Retail"),
    (r"\b(?:supply chain|logistics|freight|warehousing|procurement)\b", "Logistics & Supply Chain"),
    (r"\b(?:cybersecurity|security operations|soc|threat intelligence)\b", "Cybersecurity"),
    (r"\b(?:edtech|education|learning platform|e-learning)\b", "EdTech & Education"),
    (r"\b(?:cleantech|energy|climate tech|renewable energy|solar)\b", "CleanTech & Energy"),
    (r"\b(?:enterprise software|saas|workflow software|b2b software)\b", "Enterprise Software"),
]


class ResearchAgent:
    """Investigates individual Candidate entities and extracts source-backed Evidence.

    Features adaptive evidence gathering, cost-efficient early stopping,
    deep signal & industry extraction, and mathematical research quality evaluation.
    Does NOT decide qualification, scoring, or priority.
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
        """Execute adaptive research on one candidate, extract structured evidence, and return ResearchResult."""
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
        all_content_snippets: list[str] = []

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

        # 2. Adaptive Query Execution Loop
        all_queries = self.generate_research_queries(candidate)
        executed_queries: list[str] = []
        seen_urls: set[str] = set()
        is_saturated = False

        while len(executed_queries) < self.max_queries:
            # Check saturation criteria: identity known, leadership found, tech & need signals found, >=3 evidence with Tier 1/2
            if executed_queries:
                has_identity = website is not None or location is not None
                has_leadership = len(people) > 0
                has_tech = len(technology_signals) > 0
                has_need = len(problem_signals) > 0 or len(business_signals) > 0
                has_tier1_2 = any(e.source.reliability >= 75 for e in evidence_list)

                if has_identity and has_leadership and has_tech and has_need and len(evidence_list) >= 3 and has_tier1_2:
                    is_saturated = True
                    break

            # Select next query adaptively based on missing gaps
            if not executed_queries:
                next_query = all_queries[0]
            else:
                gap_query: str | None = None
                if not technology_signals:
                    gap_query = f'"{company_name}" "AI"'
                elif not problem_signals:
                    gap_query = f'"{company_name}" automation'
                elif not people:
                    gap_query = f'"{company_name}" engineering'
                elif not business_signals:
                    gap_query = f'"{company_name}" hiring jobs'
                else:
                    gap_query = f'"{company_name}" announcement launch funding'

                if gap_query and gap_query not in executed_queries:
                    next_query = gap_query
                else:
                    remaining = [q for q in all_queries if q not in executed_queries]
                    if not remaining:
                        break
                    next_query = remaining[0]

            executed_queries.append(next_query)

            try:
                items = self.provider.search(next_query, max_results=3)
            except Exception as exc:
                msg = f"Search query '{next_query}' failed: {exc}"
                logger.warning(msg)
                warnings.append(msg)
                items = []

            for item in items:
                item_url_str = str(item.url)
                if item_url_str in seen_urls:
                    continue
                seen_urls.add(item_url_str)

                parsed = urlparse(item_url_str)
                domain = parsed.netloc.lower().removeprefix("www.")
                content = item.content or ""
                title = item.title or ""
                published_at = self._parse_published_date(item.published_date)
                all_content_snippets.append(content)

                source_type = self._infer_source_type(
                    item_url_str,
                    domain,
                    company_name=company_name,
                    official_domain=official_domain,
                )
                reliability = self._compute_reliability(source_type)
                recency = self._compute_recency_score(published_at)

                # A. Identify official website if not already known
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
                    r"([^.?!;\n]*\b(?:AI agent|AI agents|agentic|autonomous agent|autonomous agents|LLMs?|GenAI)\b[^.?!;\n]*)",
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

        # 3. Infer Industry if not set
        if industry is None and all_content_snippets:
            full_corpus = " ".join(all_content_snippets)
            industry = self._infer_industry(full_corpus)

        # 4. Determine status based on search and evidence results
        queries_executed = len(executed_queries)
        queries_saved = max(0, self.max_queries - queries_executed)

        if warnings and not seen_urls and not evidence_list:
            status = ResearchStatus.FAILED
        elif warnings or len(evidence_list) < 2:
            status = ResearchStatus.PARTIAL
        else:
            status = ResearchStatus.COMPLETED

        # 5. Compute deterministic research quality
        quality = self.compute_research_quality(
            candidate=candidate,
            website=website,
            location=location,
            industry=industry,
            people=people,
            technology_signals=technology_signals,
            business_signals=business_signals,
            problem_signals=problem_signals,
            evidence=evidence_list,
        )

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
            research_quality=quality,
            queries_executed=queries_executed,
            queries_saved=queries_saved,
            is_saturated=is_saturated,
            researched_at=datetime.now(UTC),
        )

    @staticmethod
    def _infer_industry(corpus: str) -> str | None:
        """Deterministically map text content to an industry sector."""
        for pattern, ind in INDUSTRY_PATTERNS:
            if re.search(pattern, corpus, re.IGNORECASE):
                return ind
        return None

    def compute_research_quality(
        self,
        candidate: Candidate,
        website: HttpUrl | None,
        location: str | None,
        industry: str | None,
        people: list[DecisionMaker],
        technology_signals: list[str],
        business_signals: list[str],
        problem_signals: list[str],
        evidence: list[Evidence],
    ) -> ResearchQuality:
        """Compute mathematical research completeness, diversity, and quality scores (0.0 to 100.0)."""
        # 1. Completeness Score (100 max)
        comp = 0.0
        if website:
            comp += 20.0
        if location:
            comp += 15.0
        if industry:
            comp += 15.0
        if len(people) >= 2:
            comp += 20.0
        elif len(people) == 1:
            comp += 15.0
        if len(technology_signals) >= 1:
            comp += 15.0
        if len(problem_signals) >= 1 or len(business_signals) >= 1:
            comp += 15.0
        completeness = round(min(100.0, comp), 1)

        # 2. Evidence Diversity Score (100 max)
        if not evidence:
            diversity = 0.0
        else:
            tier1 = {SourceType.COMPANY_WEBSITE, SourceType.OFFICIAL_JOB, SourceType.OFFICIAL_ANNOUNCEMENT}
            tier2 = {SourceType.NEWS, SourceType.INDUSTRY_PUBLICATION}
            tier3 = {SourceType.LINKEDIN, SourceType.JOB_BOARD}
            tier4 = {SourceType.DIRECTORY, SourceType.SOCIAL, SourceType.OTHER}

            div = 0.0
            tiers_hit = 0
            if any(e.source.source_type in tier1 for e in evidence):
                div += 40.0
                tiers_hit += 1
            if any(e.source.source_type in tier2 for e in evidence):
                div += 30.0
                tiers_hit += 1
            if any(e.source.source_type in tier3 for e in evidence):
                div += 20.0
                tiers_hit += 1
            if any(e.source.source_type in tier4 for e in evidence):
                div += 10.0
                tiers_hit += 1

            if tiers_hit >= 2:
                div += 10.0
            diversity = round(min(100.0, div), 1)

        # 3. Recency Score (100 max)
        if evidence:
            recency = round(sum(e.recency_score for e in evidence) / len(evidence), 1)
        else:
            recency = 50.0

        # 4. Signal Depth Score (100 max)
        ev_count = len(evidence)
        if ev_count >= 5:
            depth = 100.0
        elif ev_count >= 3:
            depth = 75.0
        elif ev_count >= 2:
            depth = 55.0
        elif ev_count == 1:
            depth = 35.0
        else:
            depth = 0.0

        # 5. Overall Quality Score
        overall = round(
            0.30 * completeness + 0.25 * diversity + 0.20 * recency + 0.25 * depth,
            1,
        )

        return ResearchQuality(
            completeness_score=completeness,
            evidence_diversity_score=diversity,
            recency_score=recency,
            signal_depth_score=depth,
            overall_quality_score=overall,
        )

    def evaluate_research(self, results: list[ResearchResult]) -> ResearchEvaluation:
        """Compute aggregate intelligence evaluation metrics for the research stage."""
        cand_count = len(results)
        queries_exec = sum(r.queries_executed for r in results)
        queries_saved = sum(r.queries_saved for r in results)
        total_planned = queries_exec + queries_saved
        cost_savings = round((queries_saved / max(total_planned, 1)) * 100.0, 1)

        total_ev = sum(len(r.evidence) for r in results)
        yield_per_q = round(total_ev / max(queries_exec, 1), 2)

        sat_count = sum(1 for r in results if r.is_saturated)
        sat_rate = round((sat_count / max(cand_count, 1)) * 100.0, 1)

        tier1 = {SourceType.COMPANY_WEBSITE, SourceType.OFFICIAL_JOB, SourceType.OFFICIAL_ANNOUNCEMENT}
        tier1_ev_count = sum(1 for r in results for e in r.evidence if e.source.source_type in tier1)
        tier1_ratio = round((tier1_ev_count / max(total_ev, 1)) * 100.0, 1)

        with_dm_count = sum(1 for r in results if len(r.people) > 0)
        dm_rate = round((with_dm_count / max(cand_count, 1)) * 100.0, 1)

        qual_scores = [r.research_quality.overall_quality_score for r in results if r.research_quality]
        avg_qual = round(sum(qual_scores) / max(len(qual_scores), 1), 1) if qual_scores else 0.0

        return ResearchEvaluation(
            candidates_researched_count=cand_count,
            total_queries_executed=queries_exec,
            total_queries_saved=queries_saved,
            cost_savings_percentage=cost_savings,
            evidence_yield_per_query=yield_per_q,
            saturation_rate=sat_rate,
            tier1_source_ratio=tier1_ratio,
            decision_maker_discovery_rate=dm_rate,
            average_quality_score=avg_qual,
            evaluated_at=datetime.now(UTC),
        )

    def _extract_people(
        self, content: str, company_name: str, source_url: HttpUrl
    ) -> list[DecisionMaker]:
        """Extract high-confidence decision maker profiles without inventing data."""
        found: list[DecisionMaker] = []

        patterns = [
            (
                r"\b(?:(?:by|at)\s+)?(?:(?:Dr|Mr|Ms|Mrs)\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}),?\s+(?:is the\s+)?(?:CEO|Chief Executive Officer)\b",
                "CEO",
            ),
            (
                r"\b(?:CEO|Chief Executive Officer)\s+(?:(?:Dr|Mr|Ms|Mrs)\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
                "CEO",
            ),
            (
                r"\b(?:(?:by|at)\s+)?(?:(?:Dr|Mr|Ms|Mrs)\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}),?\s+(?:is the\s+)?(?:CTO|Chief Technology Officer)\b",
                "CTO",
            ),
            (
                r"\b(?:CTO|Chief Technology Officer)\s+(?:(?:Dr|Mr|Ms|Mrs)\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\b",
                "CTO",
            ),
            (
                r"\b(?:(?:by|at)\s+)?(?:(?:Dr|Mr|Ms|Mrs)\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}),?\s+(?:is the\s+)?(?:VP of Engineering|Vice President of Engineering)\b",
                "VP of Engineering",
            ),
            (
                r"\b(?:(?:by|at)\s+)?(?:(?:Dr|Mr|Ms|Mrs)\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}),?\s+(?:is the\s+)?(?:Head of AI|Director of AI)\b",
                "Head of AI",
            ),
            (
                r"\b(?:founded by|co-founded by|co-founders?)\s+(?:(?:Dr|Mr|Ms|Mrs)\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})",
                "Founder",
            ),
            (
                r"\b(?:(?:by|at)\s+)?(?:(?:Dr|Mr|Ms|Mrs)\.?\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}),?\s+(?:is the\s+)?(?:Founder|Co-Founder)\b",
                "Founder",
            ),
        ]

        for pat, role in patterns:
            for match in re.finditer(pat, content, re.IGNORECASE):
                raw_name = match.group(1).strip()
                name = re.sub(r"\s+", " ", raw_name).strip(" ,.-")
                name = re.sub(r"^(?:by|at|for|from|in|of|and|the|is|a|dr\.?|mr\.?|ms\.?|mrs\.?)\s+", "", name, flags=re.IGNORECASE)
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
                if 4 <= len(name) <= 35:
                    existing = next((p for p in found if p.name == name), None)
                    if existing:
                        if existing.role == "Founder" and current_role != "Founder":
                            existing.role = current_role
                    else:
                        found.append(
                            DecisionMaker(
                                name=name, role=current_role, profile_url=None
                            )
                        )

        return found

    @staticmethod
    def _domain_matches(domain: str, target: str) -> bool:
        """Check if domain matches target or is a subdomain of target."""
        return domain == target or domain.endswith("." + target)

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
        social_domains = (
            "youtube.com",
            "instagram.com",
            "twitter.com",
            "x.com",
            "facebook.com",
            "reddit.com",
            "tiktok.com",
        )
        if any(ResearchAgent._domain_matches(domain, s_d) for s_d in social_domains):
            return SourceType.SOCIAL

        # Tier 3: Professional Networks & Job Boards
        if ResearchAgent._domain_matches(domain, "linkedin.com"):
            return SourceType.LINKEDIN
        job_domains = (
            "indeed.com",
            "greenhouse.io",
            "lever.co",
            "ashbyhq.com",
            "ziprecruiter.com",
            "jobright.ai",
            "dice.com",
        )
        if any(ResearchAgent._domain_matches(domain, j) for j in job_domains) or any(
            term in url_lower
            for term in (
                "/jobs",
                "/careers",
                "jobright",
            )
        ):
            return SourceType.JOB_BOARD

        # Tier 4: Directories & Aggregators
        dir_domains = (
            "clutch.co",
            "leadiq.com",
            "g2.com",
            "zoominfo.com",
            "crunchbase.com",
            "pitchbook.com",
            "apollo.io",
        )
        if any(ResearchAgent._domain_matches(domain, d_d) for d_d in dir_domains):
            return SourceType.DIRECTORY

        # Tier 1: Official Announcements / Press Wires
        press_domains = (
            "prnewswire.com",
            "businesswire.com",
            "globenewswire.com",
            "accesswire.com",
        )
        if (
            any(ResearchAgent._domain_matches(domain, pr_d) for pr_d in press_domains)
            or "/press-releases/" in url_lower
            or "/announcements/" in url_lower
            or "/news-releases/" in url_lower
        ):
            return SourceType.OFFICIAL_ANNOUNCEMENT

        # Tier 2: Major News & Publications
        news_domains = (
            "techcrunch.com",
            "forbes.com",
            "bloomberg.com",
            "reuters.com",
            "wsj.com",
            "nytimes.com",
            "cnbc.com",
        )
        if any(ResearchAgent._domain_matches(domain, news_d) for news_d in news_domains):
            return SourceType.NEWS
        ind_domains = (
            "venturebeat.com",
            "wired.com",
            "zdnet.com",
            "informationweek.com",
        )
        if any(ResearchAgent._domain_matches(domain, ind_d) for ind_d in ind_domains):
            return SourceType.INDUSTRY_PUBLICATION

        # Tier 1: Candidate Official Company Website
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
