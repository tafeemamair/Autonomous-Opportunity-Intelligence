import re
from datetime import UTC, datetime
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import HttpUrl

from ..schemas.common import SourceType
from ..schemas.discovery import (
    Candidate,
    CandidateResearchPriority,
    DiscoveryBudget,
    DiscoveryEvaluation,
    DiscoveryPlan,
    DiscoveryStrategy,
)
from ..schemas.objective import AOIInput, BusinessObjective

LEGAL_SUFFIXES_REGEX = re.compile(
    r"(?i)\s*,?\s*\b(?:inc(?:orporated)?|llc|l\.l\.c|ltd|limited|corp(?:oration)?|co(?:mpany)?|gmbh|llp|plc|pvt\.?\s*ltd|sa|ag)\b\.?$"
)

DOMAIN_SUFFIX_REGEX = re.compile(
    r"(?i)\.(?:com|io|ai|co|org|net|tech|dev|app|xyz)$"
)

DEFAULT_STRATEGIES = {
    "hiring_signals": (
        "Find current hiring signals that indicate demand for AI, automation, workflow, agent, or engineering capabilities.",
        "hiring",
        ['"{opportunity}" "{market}" hiring', '"{opportunity}" "{market}" jobs'],
        ["hiring", "open roles", "engineering demand"],
    ),
    "product_signals": (
        "Find product launches, AI initiatives, automation announcements, or new technical capabilities that may create implementation needs.",
        "product",
        ['"{opportunity}" "{market}" launch', '"{opportunity}" "{market}" announcement'],
        ["product launch", "agent deployment", "feature release"],
    ),
    "problem_signals": (
        "Find public evidence of operational bottlenecks, repetitive workflows, scaling problems, or other automation opportunities.",
        "problem",
        ['"{problem}" "{market}" company'],
        ["workflow bottlenecks", "scaling challenges", "manual processes"],
    ),
    "growth_signals": (
        "Find growth, funding, expansion, and hiring events that may create new technical automation demand.",
        "growth",
        ['"{market}" startup funding "{opportunity}"', '"{market}" company expansion "{opportunity}"'],
        ["funding", "expansion", "team growth"],
    ),
    "technology_signals": (
        "Find companies adopting AI, agents, LLMs, or automation technologies.",
        "technology",
        ['"{technology}" "{market}" company'],
        ["AI adoption", "autonomous systems", "LLM tooling"],
    ),
}


class DiscoveryAgent:
    """Multi-signal discovery planner, candidate normalizer, and prioritizer.

    Deterministic V1 front-of-funnel engine. Employs multi-dimensional signal synthesis,
    canonical entity normalization, and priority ranking without calling LLMs.
    """

    def plan(
        self,
        aoi_input: AOIInput,
        budget: DiscoveryBudget | None = None,
    ) -> DiscoveryPlan:
        """Create an enriched, multi-signal discovery plan across all targets, bounded by DiscoveryBudget."""
        objective = aoi_input.objective
        resolved_budget = (
            budget
            or getattr(aoi_input.constraints, "discovery_budget", None)
            or DiscoveryBudget()
        )
        markets = objective.target_markets or ["global"]
        opportunities = objective.target_opportunities or ["AI automation"]
        company_types = objective.target_company_types or []

        primary_market = markets[0]
        primary_opp = opportunities[0]

        strategies: list[DiscoveryStrategy] = []
        total_queries = 0

        for name, (purpose, signal_cat, templates, expected_sigs) in DEFAULT_STRATEGIES.items():
            if total_queries >= resolved_budget.max_queries:
                break

            expanded_queries: list[str] = []
            seen_queries: set[str] = set()

            # 1. Primary queries (preserving exact query ordering for test stability)
            for template in templates:
                if total_queries + len(expanded_queries) >= resolved_budget.max_queries:
                    break
                q = template.format(
                    opportunity=primary_opp,
                    market=primary_market,
                    problem=objective.description,
                    technology=primary_opp,
                )
                if q not in seen_queries:
                    seen_queries.add(q)
                    expanded_queries.append(q)

            # 2. Additional target markets & opportunities
            for m in markets[1:]:
                for template in templates:
                    if total_queries + len(expanded_queries) >= resolved_budget.max_queries:
                        break
                    q = template.format(
                        opportunity=primary_opp,
                        market=m,
                        problem=objective.description,
                        technology=primary_opp,
                    )
                    if q not in seen_queries:
                        seen_queries.add(q)
                        expanded_queries.append(q)

            for opp in opportunities[1:]:
                for template in templates:
                    if total_queries + len(expanded_queries) >= resolved_budget.max_queries:
                        break
                    q = template.format(
                        opportunity=opp,
                        market=primary_market,
                        problem=objective.description,
                        technology=opp,
                    )
                    if q not in seen_queries:
                        seen_queries.add(q)
                        expanded_queries.append(q)

            # 3. Company-type targeted queries if specified
            for c_type in company_types:
                if total_queries + len(expanded_queries) >= resolved_budget.max_queries:
                    break
                q = f'"{c_type}" "{primary_opp}" "{primary_market}" {signal_cat}'
                if q not in seen_queries:
                    seen_queries.add(q)
                    expanded_queries.append(q)

            total_queries += len(expanded_queries)

            strategy = DiscoveryStrategy(
                name=name,
                purpose=purpose,
                query_templates=expanded_queries,
                signal_category=signal_cat,
                market=primary_market,
                opportunity=primary_opp,
                target_company_type=company_types[0] if company_types else None,
                expected_signals=list(expected_sigs),
            )
            strategies.append(strategy)

        return DiscoveryPlan(
            strategies=strategies,
            created_at=datetime.now(UTC),
            budget=resolved_budget,
        )

    @staticmethod
    def clean_company_name(raw_name: str) -> str:
        """Canonicalize company name by stripping legal suffixes, web prefixes, and extra punctuation."""
        if not raw_name:
            return ""
        name = raw_name.strip().strip("\"'“”")
        # Strip URL schemes if passed as name
        name = re.sub(r"^https?://(?:www\.)?", "", name, flags=re.IGNORECASE)
        # Strip domain TLD if passed as name (e.g. "Acme.ai" -> "Acme")
        name = DOMAIN_SUFFIX_REGEX.sub("", name)
        # Strip legal entity suffixes
        name = LEGAL_SUFFIXES_REGEX.sub("", name)
        # Clean trailing commas, dots, hyphens
        name = name.strip(" ,.-")
        # Collapse multiple whitespaces
        name = re.sub(r"\s+", " ", name).strip()
        return name or raw_name.strip()

    @staticmethod
    def clean_website_url(url: str | HttpUrl | None) -> HttpUrl | None:
        """Standardize website URL to canonical HTTPS and strip tracking params."""
        if not url:
            return None
        url_str = str(url).strip()
        if not url_str:
            return None
        if not url_str.startswith(("http://", "https://")):
            url_str = f"https://{url_str}"
        try:
            parsed = urlparse(url_str)
            domain = parsed.netloc.lower()
            if not domain:
                return None
            clean_url = f"https://{domain}{parsed.path}".rstrip("/")
            if not parsed.path or clean_url == f"https://{domain}":
                clean_url = f"https://{domain}/"
            return HttpUrl(clean_url)
        except Exception:
            return None

    @staticmethod
    def normalize_location(location: str | None) -> str | None:
        """Sanitize location and strip trailing noise."""
        if not location:
            return None
        loc = location.strip().strip(".,;")
        loc = re.sub(r"\s+", " ", loc)
        return loc if len(loc) >= 2 else None

    @staticmethod
    def _domain_matches(domain: str, target: str) -> bool:
        """Check if domain matches target or is a subdomain of target."""
        return domain == target or domain.endswith("." + target)

    @staticmethod
    def infer_source_type_from_url(url: str | HttpUrl) -> SourceType:
        """Infer source type category from URL domain and structure."""
        u_str = str(url).lower()
        parsed = urlparse(u_str)
        domain = parsed.netloc.lower().removeprefix("www.")

        social_domains = ("youtube.com", "twitter.com", "x.com", "reddit.com", "tiktok.com", "facebook.com", "instagram.com")
        if any(DiscoveryAgent._domain_matches(domain, s) for s in social_domains):
            return SourceType.SOCIAL
        if DiscoveryAgent._domain_matches(domain, "linkedin.com"):
            return SourceType.LINKEDIN
        job_domains = ("indeed.com", "greenhouse.io", "lever.co", "ashbyhq.com", "ziprecruiter.com", "jobright.ai", "dice.com")
        if any(DiscoveryAgent._domain_matches(domain, j) for j in job_domains) or any(j in u_str for j in ("/jobs", "/careers", "jobright")):
            return SourceType.JOB_BOARD
        dir_domains = ("clutch.co", "g2.com", "crunchbase.com", "pitchbook.com", "zoominfo.com", "leadiq.com", "apollo.io")
        if any(DiscoveryAgent._domain_matches(domain, d) for d in dir_domains):
            return SourceType.DIRECTORY
        news_domains = ("techcrunch.com", "forbes.com", "bloomberg.com", "reuters.com", "wsj.com", "nytimes.com", "cnbc.com")
        if any(DiscoveryAgent._domain_matches(domain, n) for n in news_domains):
            return SourceType.NEWS
        ind_domains = ("venturebeat.com", "wired.com", "zdnet.com", "informationweek.com")
        if any(DiscoveryAgent._domain_matches(domain, i) for i in ind_domains):
            return SourceType.INDUSTRY_PUBLICATION
        press_domains = ("prnewswire.com", "businesswire.com", "globenewswire.com", "accesswire.com")
        if any(DiscoveryAgent._domain_matches(domain, pr) for pr in press_domains) or any(pr in u_str for pr in ("/press-releases/", "/announcements/", "/news-releases/")):
            return SourceType.OFFICIAL_ANNOUNCEMENT
        if domain:
            return SourceType.COMPANY_WEBSITE
        return SourceType.OTHER

    def normalize_candidate(
        self,
        company_name: str,
        discovery_strategy: str,
        discovery_reason: str,
        website: str | HttpUrl | None = None,
        location: str | None = None,
        source_urls: list[str | HttpUrl] | None = None,
        objective: BusinessObjective | None = None,
    ) -> Candidate:
        """Normalize raw candidate fields into a canonical, prioritized Candidate entity."""
        display_name = (company_name or "").strip().strip("\"'“”")
        canonical_clean = self.clean_company_name(company_name)
        cleaned_website = self.clean_website_url(website)
        cleaned_location = self.normalize_location(location)

        validated_urls: list[HttpUrl] = []
        source_types: list[SourceType] = []

        raw_urls = list(source_urls or [])
        for u in raw_urls:
            val_u = self.clean_website_url(u)
            if val_u and val_u not in validated_urls:
                validated_urls.append(val_u)
                source_types.append(self.infer_source_type_from_url(val_u))

        domain = cleaned_website.host.lower().removeprefix("www.") if cleaned_website else None
        signal_cat = discovery_strategy.replace("_signals", "").replace("_signal", "")

        candidate = Candidate(
            candidate_id=f"cand_{uuid4().hex[:12]}",
            company_name=display_name,
            website=cleaned_website,
            location=cleaned_location,
            discovery_strategy=discovery_strategy,
            discovery_reason=discovery_reason,
            source_urls=validated_urls,
            source_types=source_types,
            discovered_at=datetime.now(UTC),
            normalized_name=canonical_clean.lower(),
            domain=domain,
            signal_categories=[signal_cat] if signal_cat else [],
            discovery_reasons=[discovery_reason] if discovery_reason else [],
        )

        score, priority = self.compute_priority_score(candidate, objective=objective)
        candidate.research_priority_score = score
        candidate.research_priority = priority

        return candidate

    def merge_candidates(
        self,
        existing: Candidate,
        incoming: Candidate,
        objective: BusinessObjective | None = None,
    ) -> Candidate:
        """Merge multiple discoveries of the same company, accumulating signals and updating priority without mutating inputs."""
        merged = existing.model_copy(deep=True)

        for u, st in zip(incoming.source_urls, incoming.source_types, strict=False):
            if u not in merged.source_urls:
                merged.source_urls.append(u)
                merged.source_types.append(st)

        for sig in incoming.signal_categories:
            if sig not in merged.signal_categories:
                merged.signal_categories.append(sig)

        for reason in incoming.discovery_reasons:
            if reason and reason not in merged.discovery_reasons:
                merged.discovery_reasons.append(reason)

        if not merged.website and incoming.website:
            merged.website = incoming.website
            merged.domain = incoming.domain

        if not merged.location and incoming.location:
            merged.location = incoming.location

        merged.discovery_strategy = f"{merged.discovery_strategy}, {incoming.discovery_strategy}"
        merged.discovery_reason = " | ".join(merged.discovery_reasons)

        score, priority = self.compute_priority_score(merged, objective=objective)
        merged.research_priority_score = score
        merged.research_priority = priority

        return merged

    def deduplicate_candidates(
        self,
        candidates: list[Candidate],
        objective: BusinessObjective | None = None,
    ) -> list[Candidate]:
        """Deduplicate candidates by domain or canonical company name, merging evidence without mutating input objects."""
        by_key: dict[str, Candidate] = {}

        for cand in candidates:
            cand_copy = cand.model_copy(deep=True)
            key = cand_copy.domain if cand_copy.domain else self.clean_company_name(cand_copy.company_name).lower()
            key = re.sub(r"[^a-z0-9]", "", key)
            if not key:
                key = cand_copy.candidate_id

            if key in by_key:
                by_key[key] = self.merge_candidates(by_key[key], cand_copy, objective=objective)
            else:
                by_key[key] = cand_copy

        return list(by_key.values())

    def compute_priority_score(
        self,
        candidate: Candidate,
        objective: BusinessObjective | None = None,
    ) -> tuple[float, CandidateResearchPriority]:
        """Compute deterministic research priority score (0.0 to 100.0) and priority level."""
        score = 0.0

        # 1. Multi-signal corroboration (30 max)
        unique_signals = set(candidate.signal_categories)
        if len(unique_signals) >= 3:
            score += 30.0
        elif len(unique_signals) >= 2:
            score += 25.0
        elif len(candidate.source_urls) >= 2:
            score += 15.0
        else:
            score += 10.0

        # 2. Official website / Domain presence (25 max)
        if candidate.website or candidate.domain:
            score += 25.0

        # 3. Source reliability tier (20 max)
        tier1_types = {SourceType.COMPANY_WEBSITE, SourceType.OFFICIAL_JOB, SourceType.OFFICIAL_ANNOUNCEMENT}
        tier2_types = {SourceType.NEWS, SourceType.INDUSTRY_PUBLICATION}
        if any(st in tier1_types for st in candidate.source_types):
            score += 20.0
        elif any(st in tier2_types for st in candidate.source_types):
            score += 15.0
        elif candidate.source_types:
            score += 10.0
        else:
            score += 5.0

        # 4. Signal richness in discovery reason (15 max)
        reason_text = " ".join(candidate.discovery_reasons).lower()
        keywords = ("ai", "agent", "automation", "workflow", "engineer", "pipeline", "bottleneck", "launch")
        keyword_hits = sum(1 for kw in keywords if kw in reason_text)
        if keyword_hits >= 3:
            score += 15.0
        elif keyword_hits >= 1:
            score += 10.0
        else:
            score += 5.0

        # 5. Objective & Market alignment (10 max)
        if objective:
            aligned = False
            for m in objective.target_markets:
                if m.lower() in (candidate.location or "").lower() or m.lower() in reason_text:
                    aligned = True
                    break
            for opp in objective.target_opportunities:
                if opp.lower() in reason_text:
                    aligned = True
                    break
            score += 10.0 if aligned else 5.0
        else:
            score += 10.0

        final_score = round(min(100.0, max(0.0, score)), 1)

        if final_score >= 80.0:
            priority = CandidateResearchPriority.URGENT
        elif final_score >= 65.0:
            priority = CandidateResearchPriority.HIGH
        elif final_score >= 45.0:
            priority = CandidateResearchPriority.MEDIUM
        else:
            priority = CandidateResearchPriority.LOW

        return final_score, priority

    def prioritize_candidates(
        self,
        candidates: list[Candidate],
        objective: BusinessObjective | None = None,
    ) -> list[Candidate]:
        """Rank candidates descending by deterministic research priority score."""
        for c in candidates:
            score, prio = self.compute_priority_score(c, objective=objective)
            c.research_priority_score = score
            c.research_priority = prio

        return sorted(candidates, key=lambda c: c.research_priority_score, reverse=True)

    def evaluate_discovery(
        self,
        raw_candidates_count: int,
        normalized_candidates: list[Candidate],
        plan: DiscoveryPlan | None = None,
    ) -> DiscoveryEvaluation:
        """Evaluate measurable discovery yield, consolidation, and multi-signal metrics."""
        unique_count = len(normalized_candidates)
        dedup_count = max(0, raw_candidates_count - unique_count)
        dedup_rate = round((dedup_count / max(raw_candidates_count, 1)) * 100.0, 1) if raw_candidates_count > 0 else 0.0

        multi_signal_cands = [c for c in normalized_candidates if len(set(c.signal_categories)) >= 2]
        multi_count = len(multi_signal_cands)
        multi_ratio = round((multi_count / max(unique_count, 1)) * 100.0, 1) if unique_count > 0 else 0.0

        strategy_yield: dict[str, int] = {}
        for c in normalized_candidates:
            for s in c.signal_categories:
                strategy_yield[s] = strategy_yield.get(s, 0) + 1

        all_source_types: set[SourceType] = set()
        for c in normalized_candidates:
            if c.source_types:
                all_source_types.update(c.source_types)
            elif c.source_urls:
                for u in c.source_urls:
                    all_source_types.add(self.infer_source_type_from_url(u))
            elif c.website:
                all_source_types.add(self.infer_source_type_from_url(c.website))
        channel_diversity = round(min(100.0, (len(all_source_types) / 6.0) * 100.0), 1)

        budget = plan.budget if plan else None
        total_queries = sum(len(s.query_templates) for s in plan.strategies) if plan else 0
        queries_capped = budget is not None and total_queries >= budget.max_queries

        return DiscoveryEvaluation(
            candidates_discovered_count=raw_candidates_count,
            unique_companies_count=unique_count,
            deduplication_rate=dedup_rate,
            multi_signal_candidate_count=multi_count,
            multi_signal_ratio=multi_ratio,
            strategy_yield=strategy_yield,
            channel_diversity_score=channel_diversity,
            evaluated_at=datetime.now(UTC),
            budget=budget,
            queries_executed=total_queries,
            queries_capped=queries_capped,
        )
