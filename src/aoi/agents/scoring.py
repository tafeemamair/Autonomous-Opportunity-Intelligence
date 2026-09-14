import logging
import re
from datetime import UTC, datetime

from ..schemas.common import Priority, VerificationStatus
from ..schemas.discovery import Candidate
from ..schemas.objective import BusinessObjective, Constraints, OperatorProfile
from ..schemas.qualification import FitLevel, QualificationResult, QualificationStatus
from ..schemas.research import ResearchResult
from ..schemas.scoring import ScoreBreakdown, ScoringResult
from ..schemas.verification import VerificationResult

logger = logging.getLogger(__name__)


class ScoringAgent:
    """Computes transparent, deterministic Opportunity Score, Confidence Score, and Priority.

    100% deterministic V1. Operates solely on upstream factual, qualification,
    and verification results without LLM calls or external services.
    """

    def score(
        self,
        candidate: Candidate,
        research_result: ResearchResult,
        qualification_result: QualificationResult,
        verification_result: VerificationResult,
        objective: BusinessObjective | None = None,
        operator_profile: OperatorProfile | None = None,
        constraints: Constraints | None = None,
    ) -> ScoringResult:
        """Execute deterministic scoring across all 6 dimensions."""
        scoring_reasons: list[str] = []
        scoring_warnings: list[str] = []

        # 1. Need Fit (Weight: 25%)
        need_score = self._calculate_need_fit(
            qualification_result=qualification_result,
            verification_result=verification_result,
            reasons=scoring_reasons,
        )

        # 2. Capability Fit (Weight: 20%)
        capability_score = self._calculate_capability_fit(
            operator_profile=operator_profile,
            research_result=research_result,
            qualification_result=qualification_result,
            reasons=scoring_reasons,
            warnings=scoring_warnings,
        )

        # 3. Evidence Strength (Weight: 20%)
        evidence_score = self._calculate_evidence_strength(
            research_result=research_result,
            verification_result=verification_result,
            reasons=scoring_reasons,
        )

        # 4. Timing / Urgency (Weight: 15%)
        timing_score = self._calculate_timing(
            qualification_result=qualification_result,
            verification_result=verification_result,
            constraints=constraints,
            reasons=scoring_reasons,
        )

        # 5. Commercial Potential (Weight: 10%)
        commercial_score = self._calculate_commercial_potential(
            qualification_result=qualification_result,
            research_result=research_result,
            reasons=scoring_reasons,
        )

        # 6. Accessibility (Weight: 10%)
        accessibility_score = self._calculate_accessibility(
            candidate=candidate,
            research_result=research_result,
            verification_result=verification_result,
            reasons=scoring_reasons,
        )

        # 7. Opportunity Score Calculation (Formula: 25/20/20/15/10/10)
        opportunity_score = round(
            need_score * 0.25
            + capability_score * 0.20
            + evidence_score * 0.20
            + timing_score * 0.15
            + commercial_score * 0.10
            + accessibility_score * 0.10,
            1,
        )
        opportunity_score = max(0.0, min(100.0, opportunity_score))

        # 8. Independent Confidence Score Calculation
        confidence_score = self._calculate_confidence_score(
            research_result=research_result,
            verification_result=verification_result,
            warnings=scoring_warnings,
        )

        # 9. Priority Assignment with Boundary Enforcement and Diagnostic Preservation
        priority = self._assign_priority(
            opportunity_score=opportunity_score,
            qualification_result=qualification_result,
            verification_result=verification_result,
            reasons=scoring_reasons,
        )

        scores = ScoreBreakdown(
            need_fit=round(need_score, 1),
            capability_fit=round(capability_score, 1),
            evidence_strength=round(evidence_score, 1),
            timing=round(timing_score, 1),
            commercial_potential=round(commercial_score, 1),
            accessibility=round(accessibility_score, 1),
            opportunity_score=opportunity_score,
            confidence_score=confidence_score,
            priority=priority,
        )

        return ScoringResult(
            candidate_id=candidate.candidate_id,
            company_name=candidate.company_name,
            scores=scores,
            scoring_reasons=scoring_reasons,
            scoring_warnings=scoring_warnings,
            verification_status=verification_result.verification_status,
            scored_at=datetime.now(UTC),
        )

    def _calculate_need_fit(
        self,
        qualification_result: QualificationResult,
        verification_result: VerificationResult,
        reasons: list[str],
    ) -> float:
        """Calculate Need / Problem Fit (0-100 scale)."""
        base_map = {
            FitLevel.HIGH: 90.0,
            FitLevel.MEDIUM: 70.0,
            FitLevel.LOW: 40.0,
            FitLevel.UNKNOWN: 30.0,
        }
        score = base_map.get(qualification_result.need_fit, 30.0)

        # Audit verification of need claim
        need_claims = [
            c
            for c in (
                verification_result.verified_claims
                + verification_result.partially_verified_claims
                + verification_result.unverified_claims
                + verification_result.contradicted_claims
            )
            if "engineering requirement" in c.claim.lower() or "need" in c.claim.lower()
        ]

        if need_claims:
            nc = need_claims[0]
            if nc.status == VerificationStatus.VERIFIED:
                score += 10.0
                reasons.append("Core requirement verified across multiple sources.")
            elif nc.status == VerificationStatus.PARTIALLY_VERIFIED:
                reasons.append("Core requirement supported by single source domain.")
            elif nc.status == VerificationStatus.UNVERIFIED:
                score -= 30.0
                reasons.append("Requirement claim lacked authoritative source evidence.")
            elif nc.status == VerificationStatus.CONTRADICTED:
                score -= 60.0
                reasons.append("Active requirement contradicted by recent findings.")

        return max(0.0, min(100.0, score))

    def _calculate_capability_fit(
        self,
        operator_profile: OperatorProfile | None,
        research_result: ResearchResult,
        qualification_result: QualificationResult,
        reasons: list[str],
        warnings: list[str],
    ) -> float:
        """Calculate Capability Fit. Strictly UNKNOWN (50.0) if profile is empty."""
        if (
            not operator_profile
            or (
                not operator_profile.capabilities
                and not operator_profile.portfolio_projects
                and not operator_profile.preferred_services
            )
        ):
            warnings.append("Capability fit is UNKNOWN because operator profile is empty.")
            reasons.append("Operator profile is empty; capability fit scored neutrally as UNKNOWN (50.0).")
            return 50.0

        all_operator_items = (
            operator_profile.capabilities
            + operator_profile.portfolio_projects
            + operator_profile.preferred_services
        )
        operator_tokens = set()
        for item in all_operator_items:
            for tok in re.findall(r"[a-z0-9]+", item.lower()):
                if len(tok) > 2 and tok not in ("and", "the", "for", "with"):
                    operator_tokens.add(tok)

        candidate_text = " ".join(
            research_result.technology_signals
            + research_result.problem_signals
            + qualification_result.reasons
        ).lower()

        matches = sum(1 for tok in operator_tokens if tok in candidate_text)

        if matches >= 4:
            score = 95.0
            reasons.append(f"Strong capability match ({matches} overlapping capability tokens).")
        elif matches >= 2:
            score = 80.0
            reasons.append(f"Moderate capability match ({matches} overlapping capability tokens).")
        elif matches == 1:
            score = 60.0
            reasons.append("Basic capability overlap detected.")
        else:
            score = 25.0
            reasons.append("Minimal capability overlap with operator profile.")

        return max(0.0, min(100.0, score))

    def _calculate_evidence_strength(
        self,
        research_result: ResearchResult,
        verification_result: VerificationResult,
        reasons: list[str],
    ) -> float:
        """Calculate Evidence Strength using locked deterministic formula:

        Evidence Strength = 0.40 * S_reliability + 0.30 * D_domains + 0.30 * V_status
        """
        # S_reliability: Average evidence reliability (0-100)
        if research_result.evidence:
            s_reliability = sum(ev.source.reliability for ev in research_result.evidence) / len(
                research_result.evidence
            )
        else:
            s_reliability = 0.0

        # D_domains: Independent domain score (0=0, 1=60, 2=90, >=3=100)
        dom_count = verification_result.independent_sources
        if dom_count >= 3:
            d_domains = 100.0
        elif dom_count == 2:
            d_domains = 90.0
        elif dom_count == 1:
            d_domains = 60.0
        else:
            d_domains = 0.0

        # V_status: Verification status score
        status_map = {
            VerificationStatus.VERIFIED: 100.0,
            VerificationStatus.PARTIALLY_VERIFIED: 65.0,
            VerificationStatus.UNVERIFIED: 30.0,
            VerificationStatus.CONTRADICTED: 10.0,
        }
        v_status = status_map.get(verification_result.verification_status, 30.0)

        score = (0.40 * s_reliability) + (0.30 * d_domains) + (0.30 * v_status)
        reasons.append(
            f"Evidence strength: {dom_count} independent source domain(s), avg reliability {s_reliability:.1f}."
        )
        return max(0.0, min(100.0, score))

    def _calculate_timing(
        self,
        qualification_result: QualificationResult,
        verification_result: VerificationResult,
        constraints: Constraints | None,
        reasons: list[str],
    ) -> float:
        """Calculate Timing / Urgency (0-100 scale)."""
        timing_map = {
            FitLevel.HIGH: 90.0,
            FitLevel.MEDIUM: 70.0,
            FitLevel.LOW: 40.0,
            FitLevel.UNKNOWN: 30.0,
        }
        score = timing_map.get(qualification_result.timing, 40.0)

        # Downgrade if recency window warning was triggered
        if any("recency window" in w.lower() or "stale" in w.lower() for w in verification_result.verification_warnings):
            score = min(score, 45.0)
            reasons.append("Timing penalized: evidence exceeds recency window.")
        elif any(c.status == VerificationStatus.VERIFIED for c in verification_result.verified_claims if "recency" in c.claim.lower()):
            score = min(100.0, score + 10.0)
            reasons.append("Active recency confirmed within configured window.")

        return max(0.0, min(100.0, score))

    def _calculate_commercial_potential(
        self,
        qualification_result: QualificationResult,
        research_result: ResearchResult,
        reasons: list[str],
    ) -> float:
        """Calculate Commercial Potential without fabricating dollar values."""
        base_map = {
            FitLevel.HIGH: 85.0,
            FitLevel.MEDIUM: 65.0,
            FitLevel.LOW: 35.0,
            FitLevel.UNKNOWN: 30.0,
        }
        score = base_map.get(qualification_result.commercial_relevance, 50.0)

        # Minor boost if clear business expansion signals are present
        if len(research_result.business_signals) >= 2 or len(research_result.problem_signals) >= 2:
            score = min(100.0, score + 10.0)
            reasons.append("Commercial potential supported by multiple business & expansion signals.")

        return max(0.0, min(100.0, score))

    def _calculate_accessibility(
        self,
        candidate: Candidate,
        research_result: ResearchResult,
        verification_result: VerificationResult,
        reasons: list[str],
    ) -> float:
        """Calculate Accessibility based on verifiable contact and application surfaces."""
        score = 0.0

        # Website surface
        if candidate.website:
            score += 25.0

        # Job application / ATS surface
        if any(ev.source.source_type.value in ("official_job", "job_board") for ev in research_result.evidence):
            score += 25.0
            reasons.append("Direct career/ATS application surface identified.")

        # Decision-maker surface
        if research_result.people:
            # Check if any decision maker claim was verified
            dm_verified = any(
                c.status == VerificationStatus.VERIFIED
                for c in verification_result.verified_claims
                if any(p.name and p.name.lower() in c.claim.lower() for p in research_result.people)
            )
            if dm_verified:
                score += 50.0
                reasons.append("Identified leadership contact verified across independent sources.")
            else:
                score += 25.0
                reasons.append("Leadership contact identified but not independently verified.")
        else:
            reasons.append("No decision maker contact identified; accessibility limited to organization surfaces.")

        return max(0.0, min(100.0, score))

    def _calculate_confidence_score(
        self,
        research_result: ResearchResult,
        verification_result: VerificationResult,
        warnings: list[str],
    ) -> float:
        """Calculate Confidence Score using locked deterministic formula:

        Raw Confidence = 0.35 * V_status + 0.20 * S_quality + 0.20 * D_count + 0.15 * M_coverage + 0.10 * R_consistency
        """
        # V_status: Verification quality
        v_map = {
            VerificationStatus.VERIFIED: 100.0,
            VerificationStatus.PARTIALLY_VERIFIED: 60.0,
            VerificationStatus.UNVERIFIED: 25.0,
            VerificationStatus.CONTRADICTED: 0.0,
        }
        v_status = v_map.get(verification_result.verification_status, 25.0)

        # S_quality: Average source reliability
        if research_result.evidence:
            s_quality = sum(ev.source.reliability for ev in research_result.evidence) / len(
                research_result.evidence
            )
        else:
            s_quality = 0.0

        # D_count: Independent source count score (0=0, 1=50, 2=85, >=3=100)
        dom_count = verification_result.independent_sources
        if dom_count >= 3:
            d_count = 100.0
        elif dom_count == 2:
            d_count = 85.0
        elif dom_count == 1:
            d_count = 50.0
        else:
            d_count = 0.0

        # M_coverage: Material claims verification coverage ratio
        total_claims = (
            len(verification_result.verified_claims)
            + len(verification_result.partially_verified_claims)
            + len(verification_result.unverified_claims)
            + len(verification_result.contradicted_claims)
        )
        if total_claims > 0:
            verified_count = len(verification_result.verified_claims)
            m_coverage = (verified_count / total_claims) * 100.0
        else:
            m_coverage = 0.0

        # R_consistency: Freshness and absence of warnings
        if verification_result.verification_warnings:
            r_consistency = 50.0
        else:
            r_consistency = 100.0

        raw_confidence = (
            0.35 * v_status
            + 0.20 * s_quality
            + 0.20 * d_count
            + 0.15 * m_coverage
            + 0.10 * r_consistency
        )

        # Severe Contradiction Penalty
        if verification_result.verification_status == VerificationStatus.CONTRADICTED:
            raw_confidence = min(raw_confidence * 0.15, 15.0)
            warnings.append("Severe confidence penalty applied due to verified contradiction.")

        return round(max(0.0, min(100.0, raw_confidence)), 1)

    def _assign_priority(
        self,
        opportunity_score: float,
        qualification_result: QualificationResult,
        verification_result: VerificationResult,
        reasons: list[str],
    ) -> Priority:
        """Assign Priority using locked thresholds with strict diagnostic overrides."""
        # 1. Base Priority from Opportunity Score
        if opportunity_score >= 80.0:
            base_priority = Priority.HIGH
        elif opportunity_score >= 65.0:
            base_priority = Priority.QUALIFIED
        elif opportunity_score >= 50.0:
            base_priority = Priority.WATCHLIST
        else:
            base_priority = Priority.DISCARD

        # 2. Diagnostic Override: Disqualified candidates MUST be DISCARD
        if qualification_result.qualification_status == QualificationStatus.DISQUALIFIED:
            reasons.append("Candidate disqualified in qualification stage; priority forced to DISCARD.")
            return Priority.DISCARD

        # 3. Diagnostic Override: Contradicted candidates MUST be DISCARD (never HIGH)
        if verification_result.verification_status == VerificationStatus.CONTRADICTED:
            reasons.append("Material contradiction detected in verification; priority forced to DISCARD.")
            return Priority.DISCARD

        # 4. Watchlist candidate priority capped at WATCHLIST
        if qualification_result.qualification_status == QualificationStatus.WATCHLIST:
            if base_priority in (Priority.HIGH, Priority.QUALIFIED):
                reasons.append("Candidate on WATCHLIST; priority capped at WATCHLIST.")
                return Priority.WATCHLIST

        return base_priority
