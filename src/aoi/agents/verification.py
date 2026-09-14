import logging
import re
from datetime import UTC, datetime
from urllib.parse import urlparse

from ..schemas.common import Evidence, SourceType, VerificationStatus
from ..schemas.discovery import Candidate
from ..schemas.objective import BusinessObjective, Constraints
from ..schemas.qualification import ProspectType, QualificationResult, QualificationStatus
from ..schemas.research import ResearchResult
from ..schemas.verification import VerificationClaim, VerificationResult

logger = logging.getLogger(__name__)

CONTRADICTION_PATTERNS = [
    r"\b(?:hiring freeze|freeze on hiring|paused hiring)\b",
    r"\b(?:cancelled role|canceled role|role closed|position filled|no longer hiring)\b",
    r"\b(?:laid off|layoffs|downsizing|shut down|ceasing operations|bankrupt)\b",
]

VENDOR_EVIDENCE_PATTERNS = [
    r"\b(?:ai development company|ai-first development company)\b",
    r"\b(?:ai agent development services?|ai agent development company)\b",
    r"\b(?:custom ai solutions?|custom ai development)\b",
    r"\b(?:ai transformation consulting|automation consulting)\b",
    r"\b(?:hire dedicated ai developers?|software development agency)\b",
]

PARTNER_EVIDENCE_PATTERNS = [
    r"\b(?:partner program|technology partner|integration partner|marketplace partner)\b",
    r"\b(?:partnering with clients|ecosystem partner|solution partner)\b",
]

NEED_EVIDENCE_PATTERNS = [
    r"\b(?:staff ai platform engineer|ai platform engineer|ai engineer|machine learning engineer)\b",
    r"\b(?:hiring.*ai|seeking.*engineer|open role.*ai)\b",
    r"\b(?:ai agents?|agentic systems?|autonomous agents?|llm orchestration)\b",
    r"\b(?:workflow automation|automate repetitive tasks|internal tooling)\b",
]


class VerificationAgent:
    """Independently audits and corroborates the factual claims behind a QualificationResult.

    100% deterministic V1. Evaluates source independence, reliability tiers,
    recency constraints, and contradictions without external API calls.
    """

    def verify(
        self,
        candidate: Candidate,
        research_result: ResearchResult,
        qualification_result: QualificationResult,
        objective: BusinessObjective | None = None,
        constraints: Constraints | None = None,
    ) -> VerificationResult:
        """Execute deterministic verification on qualification claims."""
        applied_constraints = constraints or Constraints()
        recency_window_days = applied_constraints.recency_days

        evidence_list = research_result.evidence
        evidence_by_id = {ev.id: ev for ev in evidence_list}
        all_unique_domains = self._get_unique_domains(evidence_list)

        verified_claims: list[VerificationClaim] = []
        partially_verified_claims: list[VerificationClaim] = []
        unverified_claims: list[VerificationClaim] = []
        contradicted_claims: list[VerificationClaim] = []
        warnings: list[str] = []

        # Check for global contradiction signals across all research evidence
        contradiction_evidence = self._find_contradiction_evidence(evidence_list)

        # 1. Audit Need Claim
        need_claim = self._verify_need_claim(
            candidate.company_name,
            evidence_list,
            qualification_result,
            recency_window_days,
            contradiction_evidence,
            warnings,
        )
        self._classify_claim(
            need_claim,
            verified_claims,
            partially_verified_claims,
            unverified_claims,
            contradicted_claims,
        )

        # 2. Audit Technology Claim
        tech_claim = self._verify_technology_claim(
            candidate.company_name,
            evidence_list,
            research_result,
            contradiction_evidence,
        )
        self._classify_claim(
            tech_claim,
            verified_claims,
            partially_verified_claims,
            unverified_claims,
            contradicted_claims,
        )

        # 3. Audit Timing Claim
        timing_claim = self._verify_timing_claim(
            candidate.company_name,
            evidence_list,
            recency_window_days,
            warnings,
        )
        self._classify_claim(
            timing_claim,
            verified_claims,
            partially_verified_claims,
            unverified_claims,
            contradicted_claims,
        )

        # 4. Audit Prospect-Type Claim
        prospect_claim = self._verify_prospect_type_claim(
            candidate.company_name,
            evidence_list,
            qualification_result.prospect_type,
            contradiction_evidence,
        )
        self._classify_claim(
            prospect_claim,
            verified_claims,
            partially_verified_claims,
            unverified_claims,
            contradicted_claims,
        )

        # 5. Audit Decision-Maker Claims (if present)
        for person in research_result.people:
            dm_claim = self._verify_decision_maker_claim(
                candidate.company_name,
                person.name or "Leadership contact",
                person.role or "Executive",
                evidence_list,
            )
            self._classify_claim(
                dm_claim,
                verified_claims,
                partially_verified_claims,
                unverified_claims,
                contradicted_claims,
            )

        # 6. Synthesize Overall Verification Status
        overall_status = self._decide_overall_status(
            qualification_result=qualification_result,
            need_claim=need_claim,
            timing_claim=timing_claim,
            prospect_claim=prospect_claim,
            tech_claim=tech_claim,
            contradicted_claims=contradicted_claims,
            warnings=warnings,
        )

        # Clean all claim evidence IDs to guarantee referential integrity
        all_claims = (
            verified_claims
            + partially_verified_claims
            + unverified_claims
            + contradicted_claims
        )
        for cl in all_claims:
            cl.supporting_evidence_ids = [
                eid for eid in cl.supporting_evidence_ids if eid in evidence_by_id
            ]
            cl.corroborating_evidence_ids = [
                eid for eid in cl.corroborating_evidence_ids if eid in evidence_by_id
            ]
            cl.contradiction_evidence_ids = [
                eid for eid in cl.contradiction_evidence_ids if eid in evidence_by_id
            ]

        return VerificationResult(
            candidate_id=candidate.candidate_id,
            company_name=candidate.company_name,
            verification_status=overall_status,
            verified_claims=verified_claims,
            partially_verified_claims=partially_verified_claims,
            unverified_claims=unverified_claims,
            contradicted_claims=contradicted_claims,
            verification_warnings=warnings,
            evidence_checked=len(evidence_list),
            independent_sources=len(all_unique_domains),
            verified_at=datetime.now(UTC),
        )

    def _verify_need_claim(
        self,
        company_name: str,
        evidence_list: list[Evidence],
        qualification_result: QualificationResult,
        recency_days: int,
        contradiction_evidence: list[Evidence],
        warnings: list[str],
    ) -> VerificationClaim:
        """Audit the core business need / hiring claim."""
        claim_text = f"{company_name} has an active AI engineering requirement or automation need."
        supporting = []
        contradiction_ids = [ev.id for ev in contradiction_evidence]

        for ev in evidence_list:
            text = f"{ev.claim} {ev.evidence_summary}".lower()
            if any(re.search(p, text, re.IGNORECASE) for p in NEED_EVIDENCE_PATTERNS):
                supporting.append(ev)

        if contradiction_evidence:
            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.CONTRADICTED,
                supporting_evidence_ids=[ev.id for ev in supporting],
                contradiction_evidence_ids=contradiction_ids,
                explanation="Reliable evidence materially contradicts the active hiring/need claim.",
            )

        if not supporting:
            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.UNVERIFIED,
                supporting_evidence_ids=[],
                explanation="No direct source evidence supports an active AI or automation need.",
            )

        # Rule E: Stale current-need evidence check
        now = datetime.now(UTC)
        recent_supporting = []
        for ev in supporting:
            if ev.source.published_at:
                if (now - ev.source.published_at).days <= recency_days:
                    recent_supporting.append(ev)
            elif ev.recency_score >= 80:
                recent_supporting.append(ev)

        if not recent_supporting and qualification_result.qualification_status == QualificationStatus.QUALIFIED:
            msg = f"Current hiring need could not be independently confirmed; evidence exceeds {recency_days}-day recency window."
            warnings.append(msg)
            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.PARTIALLY_VERIFIED,
                supporting_evidence_ids=[ev.id for ev in supporting],
                explanation=f"Historical evidence exists, but current need exceeds configured {recency_days}-day recency threshold.",
            )

        # Evaluate source independence on supporting evidence
        domains = self._get_unique_domains(supporting)
        strong_sources = [ev for ev in supporting if self._is_strong_source(ev)]

        if len(domains) >= 2 and strong_sources:
            corroborating = [ev.id for ev in supporting if self._normalize_domain(ev.source.url) != domains[0]]
            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.VERIFIED,
                supporting_evidence_ids=[ev.id for ev in supporting],
                corroborating_evidence_ids=corroborating,
                explanation=f"Corroborated across {len(domains)} independent sources including Tier 1/2 coverage.",
            )

        if strong_sources:
            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.PARTIALLY_VERIFIED,
                supporting_evidence_ids=[ev.id for ev in supporting],
                explanation="Supported by a single reliable source domain; lacks independent secondary corroboration.",
            )

        return VerificationClaim(
            claim=claim_text,
            status=VerificationStatus.UNVERIFIED,
            supporting_evidence_ids=[ev.id for ev in supporting],
            explanation="Supported only by low-quality or non-authoritative third-party source.",
        )

    def _verify_technology_claim(
        self,
        company_name: str,
        evidence_list: list[Evidence],
        research_result: ResearchResult,
        contradiction_evidence: list[Evidence],
    ) -> VerificationClaim:
        """Audit technology adoption claim. Allows historical tech claims under Adjustment 2."""
        claim_text = f"{company_name} is associated with AI/agent technology development or infrastructure."
        supporting = []

        for ev in evidence_list:
            text = f"{ev.claim} {ev.evidence_summary}".lower()
            if any(term in text for term in ("ai agent", "agentic", "llm", "genai", "autonomous", "automation")):
                supporting.append(ev)

        if not supporting and not research_result.technology_signals:
            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.UNVERIFIED,
                supporting_evidence_ids=[],
                explanation="No verifiable AI or agentic technology signals found in evidence.",
            )

        domains = self._get_unique_domains(supporting)
        strong_sources = [ev for ev in supporting if self._is_strong_source(ev)]

        if len(domains) >= 2 and strong_sources:
            corroborating = [ev.id for ev in supporting if self._normalize_domain(ev.source.url) != domains[0]]
            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.VERIFIED,
                supporting_evidence_ids=[ev.id for ev in supporting],
                corroborating_evidence_ids=corroborating,
                explanation=f"Technology adoption independently corroborated across {len(domains)} source domains.",
            )

        if strong_sources or supporting:
            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.PARTIALLY_VERIFIED,
                supporting_evidence_ids=[ev.id for ev in supporting],
                explanation="Technology presence supported by primary source evidence without secondary corroboration.",
            )

        return VerificationClaim(
            claim=claim_text,
            status=VerificationStatus.UNVERIFIED,
            supporting_evidence_ids=[ev.id for ev in supporting],
            explanation="Technology signals lack authoritative primary or secondary documentation.",
        )

    def _verify_timing_claim(
        self,
        company_name: str,
        evidence_list: list[Evidence],
        recency_days: int,
        warnings: list[str],
    ) -> VerificationClaim:
        """Audit whether signals fall within the objective's recency window."""
        claim_text = f"{company_name} key findings fall within the {recency_days}-day recency window."
        now = datetime.now(UTC)

        dated_items = [ev for ev in evidence_list if ev.source.published_at]
        recent_items = [
            ev for ev in dated_items if (now - ev.source.published_at).days <= recency_days
        ]
        high_recency_items = [ev for ev in evidence_list if ev.recency_score >= 80]

        if recent_items or high_recency_items:
            valid_supporting = recent_items or high_recency_items
            domains = self._get_unique_domains(valid_supporting)
            status = VerificationStatus.VERIFIED if len(domains) >= 2 else VerificationStatus.PARTIALLY_VERIFIED
            return VerificationClaim(
                claim=claim_text,
                status=status,
                supporting_evidence_ids=[ev.id for ev in valid_supporting],
                explanation=f"Evidence freshness confirmed within {recency_days}-day requirement.",
            )

        if dated_items:
            msg = f"The available evidence is older than the configured {recency_days}-day recency window."
            warnings.append(msg)
            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.UNVERIFIED,
                supporting_evidence_ids=[ev.id for ev in dated_items],
                explanation=f"All dated evidence exceeds the {recency_days}-day recency threshold.",
            )

        return VerificationClaim(
            claim=claim_text,
            status=VerificationStatus.PARTIALLY_VERIFIED,
            supporting_evidence_ids=[ev.id for ev in evidence_list],
            explanation="Publication dates were not provided by search results; recency is estimated.",
        )

    def _verify_prospect_type_claim(
        self,
        company_name: str,
        evidence_list: list[Evidence],
        prospect_type: ProspectType,
        contradiction_evidence: list[Evidence],
    ) -> VerificationClaim:
        """Audit the commercial classification (Customer, Vendor, Partner)."""
        claim_text = f"{company_name} is classified as {prospect_type}."
        supporting = []

        if prospect_type == ProspectType.COMPETITOR_OR_VENDOR:
            for ev in evidence_list:
                text = f"{ev.claim} {ev.evidence_summary}".lower()
                if any(re.search(p, text, re.IGNORECASE) for p in VENDOR_EVIDENCE_PATTERNS):
                    supporting.append(ev)

            domains = self._get_unique_domains(supporting)
            strong_sources = [ev for ev in supporting if self._is_strong_source(ev)]
            if len(domains) >= 2 and strong_sources:
                return VerificationClaim(
                    claim=claim_text,
                    status=VerificationStatus.VERIFIED,
                    supporting_evidence_ids=[ev.id for ev in supporting],
                    explanation="Direct evidence identifies service offerings, confirming vendor/competitor classification.",
                )
            if strong_sources:
                return VerificationClaim(
                    claim=claim_text,
                    status=VerificationStatus.PARTIALLY_VERIFIED,
                    supporting_evidence_ids=[ev.id for ev in supporting],
                    explanation="Direct evidence identifies service offerings from single source domain.",
                )

            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.UNVERIFIED,
                supporting_evidence_ids=[ev.id for ev in supporting],
                explanation="Vendor classification lacks direct reliable evidence of service sales.",
            )

        if prospect_type == ProspectType.POTENTIAL_PARTNER:
            for ev in evidence_list:
                text = f"{ev.claim} {ev.evidence_summary}".lower()
                if any(re.search(p, text, re.IGNORECASE) for p in PARTNER_EVIDENCE_PATTERNS):
                    supporting.append(ev)

            domains = self._get_unique_domains(supporting)
            strong_sources = [ev for ev in supporting if self._is_strong_source(ev)]
            if len(domains) >= 2 and strong_sources:
                return VerificationClaim(
                    claim=claim_text,
                    status=VerificationStatus.VERIFIED,
                    supporting_evidence_ids=[ev.id for ev in supporting],
                    explanation="Partner/integration program evidence confirms potential partnership classification.",
                )
            if strong_sources:
                return VerificationClaim(
                    claim=claim_text,
                    status=VerificationStatus.PARTIALLY_VERIFIED,
                    supporting_evidence_ids=[ev.id for ev in supporting],
                    explanation="Partner/integration program evidence supported by single source domain.",
                )

            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.UNVERIFIED,
                supporting_evidence_ids=[ev.id for ev in supporting],
                explanation="No explicit reliable partnership evidence found.",
            )

        if prospect_type == ProspectType.POTENTIAL_CUSTOMER:
            for ev in evidence_list:
                text = f"{ev.claim} {ev.evidence_summary}".lower()
                if any(re.search(p, text, re.IGNORECASE) for p in NEED_EVIDENCE_PATTERNS):
                    supporting.append(ev)

            domains = self._get_unique_domains(supporting)
            strong_sources = [ev for ev in supporting if self._is_strong_source(ev)]
            if len(domains) >= 2 and strong_sources:
                return VerificationClaim(
                    claim=claim_text,
                    status=VerificationStatus.VERIFIED,
                    supporting_evidence_ids=[ev.id for ev in supporting],
                    explanation="Internal operational or technical hiring signals support customer opportunity status.",
                )
            if strong_sources:
                return VerificationClaim(
                    claim=claim_text,
                    status=VerificationStatus.PARTIALLY_VERIFIED,
                    supporting_evidence_ids=[ev.id for ev in supporting],
                    explanation="Internal operational or technical hiring signals supported by single source domain.",
                )

            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.UNVERIFIED,
                supporting_evidence_ids=[ev.id for ev in supporting],
                explanation="Customer classification lacks verified internal hiring or operational signals.",
            )

        return VerificationClaim(
            claim=claim_text,
            status=VerificationStatus.UNVERIFIED,
            supporting_evidence_ids=[],
            explanation="Prospect type is undetermined based on available findings.",
        )

    def _verify_decision_maker_claim(
        self,
        company_name: str,
        person_name: str,
        role: str,
        evidence_list: list[Evidence],
    ) -> VerificationClaim:
        """Audit leadership identity claim."""
        claim_text = f"{person_name} is identified as {role} at {company_name}."
        supporting = []

        for ev in evidence_list:
            text = f"{ev.claim} {ev.evidence_summary}".lower()
            if person_name.lower() in text:
                supporting.append(ev)

        if not supporting:
            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.UNVERIFIED,
                supporting_evidence_ids=[],
                explanation="No evidence corroborates the identity or role of this person.",
            )

        domains = self._get_unique_domains(supporting)
        strong_sources = [ev for ev in supporting if self._is_strong_source(ev)]

        if len(domains) >= 2 and strong_sources:
            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.VERIFIED,
                supporting_evidence_ids=[ev.id for ev in supporting],
                explanation=f"Leadership role corroborated across {len(domains)} independent sources.",
            )

        if strong_sources:
            return VerificationClaim(
                claim=claim_text,
                status=VerificationStatus.PARTIALLY_VERIFIED,
                supporting_evidence_ids=[ev.id for ev in supporting],
                explanation="Leadership role supported by primary source without secondary corroboration.",
            )

        return VerificationClaim(
            claim=claim_text,
            status=VerificationStatus.UNVERIFIED,
            supporting_evidence_ids=[ev.id for ev in supporting],
            explanation="Identified only in low-reliability aggregator or directory listing.",
        )

    def _decide_overall_status(
        self,
        qualification_result: QualificationResult,
        need_claim: VerificationClaim,
        timing_claim: VerificationClaim,
        prospect_claim: VerificationClaim,
        tech_claim: VerificationClaim,
        contradicted_claims: list[VerificationClaim],
        warnings: list[str],
    ) -> VerificationStatus:
        """Derive overall VerificationStatus based on material claims (Adjustment 3)."""
        # Rule D: Any material contradiction -> CONTRADICTED
        if contradicted_claims or need_claim.status == VerificationStatus.CONTRADICTED:
            return VerificationStatus.CONTRADICTED

        # Case 1: Prospect was DISQUALIFIED (e.g. Vendor)
        if qualification_result.qualification_status == QualificationStatus.DISQUALIFIED:
            if prospect_claim.status in (VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED):
                return prospect_claim.status
            return VerificationStatus.UNVERIFIED

        # Case 2: Prospect was INSUFFICIENT_EVIDENCE
        if qualification_result.qualification_status == QualificationStatus.INSUFFICIENT_EVIDENCE:
            return VerificationStatus.UNVERIFIED

        # Case 3: Prospect was QUALIFIED
        if qualification_result.qualification_status == QualificationStatus.QUALIFIED:
            # If the core need is UNVERIFIED, the opportunity cannot be verified or partially verified
            if need_claim.status == VerificationStatus.UNVERIFIED:
                return VerificationStatus.UNVERIFIED

            material_statuses = [need_claim.status, prospect_claim.status]

            # If need or timing is unverified or downgraded, overall status cannot be fully VERIFIED
            if any(st == VerificationStatus.UNVERIFIED for st in material_statuses) or timing_claim.status == VerificationStatus.UNVERIFIED:
                if any(st in (VerificationStatus.VERIFIED, VerificationStatus.PARTIALLY_VERIFIED) for st in material_statuses):
                    return VerificationStatus.PARTIALLY_VERIFIED
                return VerificationStatus.UNVERIFIED

            if all(st == VerificationStatus.VERIFIED for st in material_statuses) and timing_claim.status == VerificationStatus.VERIFIED:
                return VerificationStatus.VERIFIED

            return VerificationStatus.PARTIALLY_VERIFIED

        # Case 4: Prospect was WATCHLIST
        if need_claim.status == VerificationStatus.VERIFIED or tech_claim.status == VerificationStatus.VERIFIED:
            return VerificationStatus.PARTIALLY_VERIFIED

        return VerificationStatus.UNVERIFIED

    def _find_contradiction_evidence(self, evidence_list: list[Evidence]) -> list[Evidence]:
        """Detect evidence items indicating material contradiction."""
        contradictions = []
        for ev in evidence_list:
            text = f"{ev.claim} {ev.evidence_summary}".lower()
            if any(re.search(pat, text, re.IGNORECASE) for pat in CONTRADICTION_PATTERNS):
                contradictions.append(ev)
        return contradictions

    @staticmethod
    def _normalize_domain(url: str) -> str:
        """Extract and normalize domain to enforce source independence."""
        try:
            netloc = urlparse(str(url)).netloc.lower()
            return netloc.removeprefix("www.").split(":")[0]
        except Exception:
            return ""

    def _get_unique_domains(self, evidence_list: list[Evidence]) -> list[str]:
        """Collect list of distinct normalized source domains."""
        domains: list[str] = []
        for ev in evidence_list:
            dom = self._normalize_domain(str(ev.source.url))
            if dom and dom not in domains:
                domains.append(dom)
        return domains

    @staticmethod
    def _is_strong_source(ev: Evidence) -> bool:
        """Check if source is Tier 1 or Tier 2 (or reliability >= 75)."""
        return (
            ev.source.source_type
            in (
                SourceType.COMPANY_WEBSITE,
                SourceType.OFFICIAL_JOB,
                SourceType.OFFICIAL_ANNOUNCEMENT,
                SourceType.REGULATORY,
                SourceType.NEWS,
                SourceType.INDUSTRY_PUBLICATION,
            )
            or ev.source.reliability >= 75
        )

    @staticmethod
    def _classify_claim(
        claim: VerificationClaim,
        verified: list[VerificationClaim],
        partially_verified: list[VerificationClaim],
        unverified: list[VerificationClaim],
        contradicted: list[VerificationClaim],
    ) -> None:
        """Append claim to its corresponding verification list."""
        match claim.status:
            case VerificationStatus.VERIFIED:
                verified.append(claim)
            case VerificationStatus.PARTIALLY_VERIFIED:
                partially_verified.append(claim)
            case VerificationStatus.UNVERIFIED:
                unverified.append(claim)
            case VerificationStatus.CONTRADICTED:
                contradicted.append(claim)
