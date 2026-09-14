import logging
import re
from datetime import UTC, datetime

from ..schemas.discovery import Candidate
from ..schemas.objective import BusinessObjective, Constraints, OperatorProfile
from ..schemas.qualification import FitLevel, ProspectType, QualificationResult, QualificationStatus
from ..schemas.research import ResearchResult, ResearchStatus

logger = logging.getLogger(__name__)

# Patterns identifying agencies, consultancies, or software vendors selling AI services
VENDOR_PATTERNS = [
    r"\b(?:ai development company|ai-first development company)\b",
    r"\b(?:ai agent development services?|ai agent development company)\b",
    r"\b(?:custom ai solutions?|custom ai development)\b",
    r"\b(?:ai transformation consulting|automation consulting)\b",
    r"\b(?:ai development services?|agentic ai solutions & services)\b",
    r"\b(?:software development agency|ai agency)\b",
    r"\b(?:hire dedicated ai developers?|top \d+ ai agent development)\b",
]

# Patterns identifying potential platform or ecosystem partners
PARTNER_PATTERNS = [
    r"\b(?:partner program|technology partner|integration partner)\b",
    r"\b(?:marketplace partner|ecosystem partner|solution partner)\b",
    r"\b(?:partnering with clients|co-sell partner|strategic alliance)\b",
]

# High-signal problem/need patterns
STRONG_NEED_PATTERNS = [
    r"\b(?:staff ai platform engineer|ai platform engineer|ai engineer|machine learning engineer)\b",
    r"\b(?:hiring.*ai|seeking.*engineer|open role.*ai)\b",
    r"\b(?:ai agents?|agentic systems?|autonomous agents?|llm orchestration)\b",
    r"\b(?:workflow automation|automate repetitive tasks|internal tooling)\b",
]


class QualificationAgent:
    """Evaluates Candidate research evidence against business objective and operator profile.

    Produces structured, explainable QualificationResult records with strict
    evidence traceability. Does NOT calculate final Opportunity or Confidence scores.
    """

    def qualify(
        self,
        candidate: Candidate,
        research_result: ResearchResult,
        objective: BusinessObjective | None = None,
        operator_profile: OperatorProfile | None = None,
        constraints: Constraints | None = None,
    ) -> QualificationResult:
        """Deterministically qualify a candidate based on research evidence."""
        profile = operator_profile or OperatorProfile()
        applied_constraints = constraints or Constraints()

        reasons: list[str] = []
        supporting_evidence_ids: list[str] = []
        disqualification_reasons: list[str] = []
        warnings: list[str] = list(research_result.warnings)

        evidence_list = research_result.evidence
        evidence_by_id = {ev.id: ev for ev in evidence_list}

        # 1. Prospect-Type Detection
        prospect_type = self._detect_prospect_type(research_result, supporting_evidence_ids)

        # 2. Need / Problem Fit Assessment
        need_fit, need_reasons = self._assess_need_fit(research_result, supporting_evidence_ids)
        reasons.extend(need_reasons)

        # 3. Capability Fit Assessment (Strict constraint: Empty capabilities -> UNKNOWN)
        capability_fit, cap_reasons = self._assess_capability_fit(research_result, profile)
        reasons.extend(cap_reasons)

        # 4. Commercial Relevance Assessment
        commercial_relevance, comm_reasons = self._assess_commercial_relevance(
            research_result, supporting_evidence_ids
        )
        reasons.extend(comm_reasons)

        # 5. Timing Assessment (Against 90-day constraint)
        timing, timing_reasons = self._assess_timing(
            research_result, applied_constraints.recency_days, supporting_evidence_ids
        )
        reasons.extend(timing_reasons)

        # 6. Evidence Sufficiency Assessment
        evidence_sufficiency = self._assess_evidence_sufficiency(research_result)

        # 7. Final Qualification Status Decision
        qualification_status = self._decide_status(
            prospect_type=prospect_type,
            need_fit=need_fit,
            commercial_relevance=commercial_relevance,
            evidence_sufficiency=evidence_sufficiency,
            timing=timing,
            research_status=research_result.research_status,
            disqualification_reasons=disqualification_reasons,
        )

        # Ensure supporting_evidence_ids are unique and strictly exist in evidence_list
        valid_supporting_ids = [
            ev_id for ev_id in dict.fromkeys(supporting_evidence_ids) if ev_id in evidence_by_id
        ]

        return QualificationResult(
            candidate_id=candidate.candidate_id,
            company_name=candidate.company_name,
            qualification_status=qualification_status,
            prospect_type=prospect_type,
            need_fit=need_fit,
            capability_fit=capability_fit,
            commercial_relevance=commercial_relevance,
            timing=timing,
            evidence_sufficiency=evidence_sufficiency,
            reasons=reasons,
            supporting_evidence_ids=valid_supporting_ids,
            disqualification_reasons=disqualification_reasons,
            warnings=warnings,
            qualified_at=datetime.now(UTC),
        )

    def _detect_prospect_type(
        self, research: ResearchResult, supporting_ids: list[str]
    ) -> ProspectType:
        """Classify prospect type: COMPETITOR_OR_VENDOR, POTENTIAL_PARTNER, POTENTIAL_CUSTOMER, or UNKNOWN."""
        # A. Check for Competitor / Vendor Signals
        for ev in research.evidence:
            text = f"{ev.claim} {ev.evidence_summary}".lower()
            if any(re.search(pat, text, re.IGNORECASE) for pat in VENDOR_PATTERNS):
                supporting_ids.append(ev.id)
                return ProspectType.COMPETITOR_OR_VENDOR

        # Also check technology and business signals for vendor markers
        for sig in research.technology_signals + research.business_signals:
            if any(re.search(pat, sig, re.IGNORECASE) for pat in VENDOR_PATTERNS):
                # Associate with any evidence mentioning services
                for ev in research.evidence:
                    if "service" in ev.claim.lower() or "development" in ev.claim.lower():
                        supporting_ids.append(ev.id)
                return ProspectType.COMPETITOR_OR_VENDOR

        # B. Check for Partner Signals
        for ev in research.evidence:
            text = f"{ev.claim} {ev.evidence_summary}".lower()
            if any(re.search(pat, text, re.IGNORECASE) for pat in PARTNER_PATTERNS):
                supporting_ids.append(ev.id)
                return ProspectType.POTENTIAL_PARTNER

        # C. Check for Customer Signals (hiring engineering or needing internal automation)
        has_customer_need = False
        for ev in research.evidence:
            text = f"{ev.claim} {ev.evidence_summary}".lower()
            if any(re.search(pat, text, re.IGNORECASE) for pat in STRONG_NEED_PATTERNS):
                has_customer_need = True
                supporting_ids.append(ev.id)

        if has_customer_need or research.problem_signals:
            return ProspectType.POTENTIAL_CUSTOMER

        # If signals are generic or neutral
        return ProspectType.UNKNOWN

    def _assess_need_fit(
        self, research: ResearchResult, supporting_ids: list[str]
    ) -> tuple[FitLevel, list[str]]:
        """Assess the presence and credibility of AI agent / automation needs."""
        reasons: list[str] = []
        matching_evidence = []

        for ev in research.evidence:
            text = f"{ev.claim} {ev.evidence_summary}".lower()
            if any(re.search(pat, text, re.IGNORECASE) for pat in STRONG_NEED_PATTERNS):
                matching_evidence.append(ev)
                supporting_ids.append(ev.id)

        if matching_evidence:
            reasons.append(
                f"Evidence indicates active need/demand related to AI agents or engineering ({len(matching_evidence)} source items)."
            )
            return FitLevel.HIGH, reasons

        if research.problem_signals:
            reasons.append("Identified operational or workflow automation signals in research.")
            return FitLevel.MEDIUM, reasons

        if research.technology_signals:
            reasons.append("General AI technology activity identified without specific engineering need.")
            return FitLevel.LOW, reasons

        return FitLevel.UNKNOWN, ["Insufficient evidence of AI or automation need."]

    def _assess_capability_fit(
        self, research: ResearchResult, profile: OperatorProfile
    ) -> tuple[FitLevel, list[str]]:
        """Assess compatibility with the operator's capabilities. Never fabricate when profile is empty."""
        reasons: list[str] = []

        if not profile.capabilities:
            return FitLevel.UNKNOWN, ["Operator profile has no configured capabilities; capability fit is unassessed."]

        corpus = " ".join(
            research.technology_signals + research.problem_signals + [ev.claim for ev in research.evidence]
        ).lower()

        matched_caps = [cap for cap in profile.capabilities if cap.lower() in corpus]

        if len(matched_caps) >= 2:
            reasons.append(f"Strong capability match with operator skills: {', '.join(matched_caps)}.")
            return FitLevel.HIGH, reasons
        if len(matched_caps) == 1:
            reasons.append(f"Plausible capability match with operator skill: {matched_caps[0]}.")
            return FitLevel.MEDIUM, reasons

        reasons.append("No direct overlap between candidate signals and operator capabilities.")
        return FitLevel.LOW, reasons

    def _assess_commercial_relevance(
        self, research: ResearchResult, supporting_ids: list[str]
    ) -> tuple[FitLevel, list[str]]:
        """Assess whether a plausible commercial engagement opportunity exists."""
        reasons: list[str] = []
        hiring_roles = []

        for ev in research.evidence:
            claim_lower = ev.claim.lower()
            if any(k in claim_lower for k in ("hiring", "open role", "seeking", "careers", "engineer")):
                hiring_roles.append(ev)
                supporting_ids.append(ev.id)

        if hiring_roles:
            reasons.append("Commercial relevance supported by active technical hiring signals.")
            return FitLevel.HIGH, reasons

        if research.problem_signals and research.technology_signals:
            reasons.append("Commercial relevance supported by active AI systems development.")
            return FitLevel.MEDIUM, reasons

        return FitLevel.LOW, ["No clear commercial need or hiring signal established in research."]

    def _assess_timing(
        self, research: ResearchResult, recency_constraint_days: int, supporting_ids: list[str]
    ) -> tuple[FitLevel, list[str]]:
        """Assess timing based on evidence recency relative to objective constraints."""
        reasons: list[str] = []
        if not research.evidence:
            return FitLevel.UNKNOWN, ["No evidence available to evaluate timing."]

        now = datetime.now(UTC)
        recent_items = []
        old_items = []

        for ev in research.evidence:
            pub_date = ev.source.published_at
            if pub_date:
                days_old = (now - pub_date).days
                if days_old <= recency_constraint_days:
                    recent_items.append(ev)
                    supporting_ids.append(ev.id)
                else:
                    old_items.append(ev)
            elif ev.recency_score >= 80:
                recent_items.append(ev)
                supporting_ids.append(ev.id)

        if recent_items:
            reasons.append(
                f"Timing is favorable: evidence contains signals within the {recency_constraint_days}-day window."
            )
            return FitLevel.HIGH, reasons

        if old_items:
            reasons.append(
                f"Evidence exceeds the {recency_constraint_days}-day recency threshold; timing may be stale."
            )
            return FitLevel.LOW, reasons

        reasons.append("Evidence timing is unverified or dates are not published.")
        return FitLevel.MEDIUM, reasons

    def _assess_evidence_sufficiency(self, research: ResearchResult) -> FitLevel:
        """Evaluate whether research findings provide a sufficient factual basis."""
        if research.research_status == ResearchStatus.FAILED or not research.evidence:
            return FitLevel.LOW

        reliable_evidence = [ev for ev in research.evidence if ev.source.reliability >= 60]
        if len(reliable_evidence) >= 2:
            return FitLevel.HIGH
        if len(research.evidence) >= 1:
            return FitLevel.MEDIUM
        return FitLevel.LOW

    def _decide_status(
        self,
        prospect_type: ProspectType,
        need_fit: FitLevel,
        commercial_relevance: FitLevel,
        evidence_sufficiency: FitLevel,
        timing: FitLevel,
        research_status: ResearchStatus,
        disqualification_reasons: list[str],
    ) -> QualificationStatus:
        """Synthesize dimensions into a deterministic QualificationStatus."""
        # 1. Obvious Competitor / Vendor -> DISQUALIFIED
        if prospect_type == ProspectType.COMPETITOR_OR_VENDOR:
            disqualification_reasons.append(
                "Disqualified as customer prospect: company is identified as an AI development vendor/competitor."
            )
            return QualificationStatus.DISQUALIFIED

        # 2. Failed research or insufficient evidence
        if research_status == ResearchStatus.FAILED or evidence_sufficiency == FitLevel.LOW:
            return QualificationStatus.INSUFFICIENT_EVIDENCE

        # 3. Strong Need + Commercial Relevance + Sufficient Evidence -> QUALIFIED
        if (
            need_fit in (FitLevel.HIGH, FitLevel.MEDIUM)
            and commercial_relevance in (FitLevel.HIGH, FitLevel.MEDIUM)
            and evidence_sufficiency in (FitLevel.HIGH, FitLevel.MEDIUM)
            and timing != FitLevel.LOW
        ):
            return QualificationStatus.QUALIFIED

        # 4. Potential partner or general AI presence without commercial customer fit -> WATCHLIST
        if need_fit == FitLevel.LOW or timing == FitLevel.LOW or prospect_type == ProspectType.POTENTIAL_PARTNER:
            return QualificationStatus.WATCHLIST

        return QualificationStatus.WATCHLIST
