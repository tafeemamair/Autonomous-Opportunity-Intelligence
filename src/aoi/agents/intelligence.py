from datetime import UTC, datetime
from typing import Sequence

from ..schemas.common import Priority, VerificationStatus
from ..schemas.discovery import Candidate
from ..schemas.intelligence import (
    IntelligenceQuality,
    IntelligenceQualityResult,
    OpportunityNarrative,
)
from ..schemas.objective import BusinessObjective, Constraints, OperatorProfile
from ..schemas.qualification import FitLevel, ProspectType, QualificationResult, QualificationStatus
from ..schemas.research import ResearchResult
from ..schemas.scoring import ScoringResult
from ..schemas.verification import VerificationResult


class IntelligenceQualityAgent:
    """Evaluates intelligence quality and opportunity depth for scored candidates.

    100% deterministic V1 layer. Does NOT search/browse/call external APIs,
    call LLMs, discover candidates, create new evidence, or alter upstream
    Opportunity Score, Confidence Score, Priority, Qualification Status,
    or Verification Status.
    """

    def evaluate(
        self,
        candidate: Candidate | None,
        research_result: ResearchResult | None,
        qualification_result: QualificationResult | None,
        verification_result: VerificationResult | None,
        scoring_result: ScoringResult | None,
        objective: BusinessObjective | None = None,
        operator_profile: OperatorProfile | None = None,
        constraints: Constraints | None = None,
        run_warnings: Sequence[str] | None = None,
        generated_at: datetime | None = None,
    ) -> IntelligenceQualityResult:
        """Execute deterministic evaluation across Completeness, Evidence Coverage, Consistency, and Actionability."""
        now = generated_at or datetime.now(UTC)
        candidate_id = (
            (scoring_result and scoring_result.candidate_id)
            or (candidate and candidate.candidate_id)
            or "unknown"
        )
        company_name = (
            (scoring_result and scoring_result.company_name)
            or (candidate and candidate.company_name)
            or "Unknown Company"
        )

        quality_warnings: list[str] = []
        strengths: list[str] = []
        weaknesses: list[str] = []

        # 1. Completeness Calculation (Core: 60 pts, Decision Usefulness: 40 pts)
        completeness_score = self._calculate_completeness(
            candidate=candidate,
            research=research_result,
            qualification=qualification_result,
            verification=verification_result,
            scoring=scoring_result,
            operator_profile=operator_profile,
        )

        # 2. Evidence Coverage Calculation (Weights: Need 30%, Tech 15%, Timing 20%, Commercial 20%, Prospect Type 15%)
        evidence_coverage_score, covered_claims, total_claims = self._calculate_evidence_coverage(
            research=research_result,
            verification=verification_result,
            qualification=qualification_result,
            warnings=quality_warnings,
            strengths=strengths,
            weaknesses=weaknesses,
        )

        # 3. Consistency Calculation (Weights: Prospect-type 25%, Need 30%, Timing 20%, Verification/scoring 25%)
        consistency_score = self._calculate_consistency(
            candidate=candidate,
            research=research_result,
            qualification=qualification_result,
            verification=verification_result,
            scoring=scoring_result,
            warnings=quality_warnings,
            weaknesses=weaknesses,
        )

        # 4. Actionability Calculation (Components: Problem 25%, Evidence 20%, Fit 20%, Timing 10%, Next step 15%, Accessibility 10%)
        actionability_score = self._calculate_actionability(
            candidate=candidate,
            research=research_result,
            qualification=qualification_result,
            verification=verification_result,
            scoring=scoring_result,
            operator_profile=operator_profile,
            strengths=strengths,
            weaknesses=weaknesses,
        )

        # Overall Quality Score
        # Quality Score = Completeness × 0.25 + Evidence Coverage × 0.30 + Consistency × 0.20 + Actionability × 0.25
        raw_quality = (
            completeness_score * 0.25
            + evidence_coverage_score * 0.30
            + consistency_score * 0.20
            + actionability_score * 0.25
        )
        quality_score = round(raw_quality, 1)

        # Signal Synthesis
        converging_signals = self._synthesize_signals(
            research=research_result,
            verification=verification_result,
            strengths=strengths,
        )

        # Opportunity Narrative
        narrative = self._build_narrative(
            candidate=candidate,
            research=research_result,
            qualification=qualification_result,
            verification=verification_result,
            scoring=scoring_result,
            operator_profile=operator_profile,
        )

        # Aggregate warnings for IntelligenceQuality
        all_quality_warnings = list(quality_warnings)
        if completeness_score < 50.0:
            all_quality_warnings.append("Low intelligence completeness; material candidate attributes missing.")
        if evidence_coverage_score < 50.0:
            all_quality_warnings.append("Low evidence coverage; multiple material claims lack verified support.")
        if consistency_score < 70.0:
            all_quality_warnings.append("Intelligence consistency gaps detected between pipeline stages.")

        intelligence_quality = IntelligenceQuality(
            completeness_score=completeness_score,
            evidence_coverage_score=evidence_coverage_score,
            consistency_score=consistency_score,
            actionability_score=actionability_score,
            quality_score=quality_score,
            warnings=all_quality_warnings,
        )

        return IntelligenceQualityResult(
            candidate_id=candidate_id,
            company_name=company_name,
            quality=intelligence_quality,
            narrative=narrative,
            strengths=strengths,
            weaknesses=weaknesses,
            converging_signals=converging_signals,
            material_claims_covered=covered_claims,
            material_claims_total=total_claims,
            quality_warnings=all_quality_warnings,
            generated_at=now,
        )

    def _calculate_completeness(
        self,
        candidate: Candidate | None,
        research: ResearchResult | None,
        qualification: QualificationResult | None,
        verification: VerificationResult | None,
        scoring: ScoringResult | None,
        operator_profile: OperatorProfile | None,
    ) -> float:
        """Calculate completeness score out of 100 points (60 core + 40 decision usefulness)."""
        # Core fields (60 points)
        # 1. Company/candidate identity: 10 pts
        c_identity = 0.0
        if candidate and candidate.company_name and candidate.candidate_id:
            if candidate.location or (research and (research.location or research.industry)):
                c_identity = 1.0
            else:
                c_identity = 0.5

        # 2. Identified problem/need: 15 pts
        c_problem = 0.0
        if research and research.problem_signals:
            c_problem = 1.0
        elif qualification and qualification.reasons:
            c_problem = 0.5

        # 3. Supporting evidence: 15 pts
        c_evidence = 0.0
        if research and research.evidence:
            c_evidence = 1.0 if len(research.evidence) >= 2 else 0.5
        elif verification and (verification.verified_claims or verification.partially_verified_claims):
            c_evidence = 0.5

        # 4. Qualification status: 5 pts
        c_qual = 0.0
        if qualification:
            c_qual = 1.0 if qualification.qualification_status != QualificationStatus.INSUFFICIENT_EVIDENCE else 0.5

        # 5. Verification status: 5 pts
        c_ver = 0.0
        if verification:
            c_ver = 1.0 if verification.verification_status != VerificationStatus.UNVERIFIED else 0.5

        # 6. Opportunity score + priority: 5 pts
        c_score = 0.0
        if scoring:
            c_score = 1.0 if scoring.scores.opportunity_score > 0 else 0.5

        # 7. Confidence score: 5 pts
        c_conf = 0.0
        if scoring:
            c_conf = 1.0 if scoring.scores.confidence_score >= 50.0 else 0.5

        core_score = (
            c_identity * 10
            + c_problem * 15
            + c_evidence * 15
            + c_qual * 5
            + c_ver * 5
            + c_score * 5
            + c_conf * 5
        )

        # Decision usefulness fields (40 points)
        # 1. Timing information: 10 pts
        d_timing = 0.0
        has_dates = bool(research and any(e.source.published_at for e in research.evidence if e.source))
        if qualification and qualification.timing == FitLevel.HIGH and has_dates:
            d_timing = 1.0
        elif has_dates or (qualification and qualification.timing in (FitLevel.HIGH, FitLevel.MEDIUM)):
            d_timing = 0.5

        # 2. Potential solution: 8 pts
        d_solution = 0.0
        if (research and research.problem_signals) and (operator_profile and operator_profile.capabilities):
            d_solution = 1.0
        elif research and research.problem_signals:
            d_solution = 0.5

        # 3. Capability/fit explanation: 7 pts
        d_fit = 0.0
        if qualification and qualification.capability_fit in (FitLevel.HIGH, FitLevel.MEDIUM):
            d_fit = 1.0 if (operator_profile and operator_profile.capabilities) else 0.5

        # 4. Scoring reasons: 5 pts
        d_reasons = 0.0
        if scoring and scoring.scoring_reasons:
            d_reasons = 1.0 if len(scoring.scoring_reasons) >= 2 else 0.5

        # 5. Recommended human action: 5 pts
        d_action = 1.0 if scoring else 0.0

        # 6. Website: 3 pts
        d_website = 0.0
        if (research and research.website) or (candidate and candidate.website):
            d_website = 1.0

        # 7. Decision maker: 2 pts
        d_dm = 0.0
        if research and research.people:
            first_person = research.people[0]
            if first_person.name and first_person.role:
                d_dm = 1.0
            elif first_person.name or first_person.role:
                d_dm = 0.5

        decision_score = (
            d_timing * 10
            + d_solution * 8
            + d_fit * 7
            + d_reasons * 5
            + d_action * 5
            + d_website * 3
            + d_dm * 2
        )

        return round(core_score + decision_score, 1)

    def _calculate_evidence_coverage(
        self,
        research: ResearchResult | None,
        verification: VerificationResult | None,
        qualification: QualificationResult | None,
        warnings: list[str],
        strengths: list[str],
        weaknesses: list[str],
    ) -> tuple[float, int, int]:
        """Calculate evidence coverage score out of 100 with dynamic normalization and contradiction detection."""
        # Material categories and base weights:
        # Need: 30%, Technology: 15%, Timing: 20%, Commercial: 20%, Prospect type: 15%
        categories: dict[str, float] = {
            "need": 0.30,
            "technology": 0.15,
            "timing": 0.20,
            "commercial": 0.20,
            "prospect_type": 0.15,
        }

        # Check for contradicted claims across verification result
        contradicted_claims = list(verification.contradicted_claims) if verification else []
        for c in contradicted_claims:
            contradiction_msg = f"Material evidence contradiction detected for {c.claim}."
            warnings.append(contradiction_msg)
            weaknesses.append(contradiction_msg)

        # Deduplicate sources to avoid gaming
        unique_sources: set[str] = set()
        if research and research.evidence:
            for ev in research.evidence:
                if ev.source and ev.source.url:
                    unique_sources.add(str(ev.source.url).rstrip("/"))

        if len(unique_sources) >= 2:
            strengths.append(f"Supported by {len(unique_sources)} independent unique evidence sources.")

        # Helper to grade coverage for a category: 100, 60, 30, or 0
        scores: dict[str, float] = {}

        # 1. Need / Problem
        has_need_contradiction = any(
            "need" in c.claim.lower() or "problem" in c.claim.lower() for c in contradicted_claims
        )
        if has_need_contradiction:
            scores["need"] = 0.0
        elif verification and any(
            "need" in c.claim.lower() or "problem" in c.claim.lower() for c in verification.verified_claims
        ):
            scores["need"] = 100.0
        elif verification and any(
            "need" in c.claim.lower() or "problem" in c.claim.lower() for c in verification.partially_verified_claims
        ):
            scores["need"] = 60.0
        elif research and research.problem_signals and research.evidence:
            scores["need"] = 30.0
        else:
            scores["need"] = 0.0

        # 2. Technology / Capability
        has_tech_contradiction = any(
            "tech" in c.claim.lower() or "ai" in c.claim.lower() for c in contradicted_claims
        )
        if has_tech_contradiction:
            scores["technology"] = 0.0
        elif verification and any(
            "tech" in c.claim.lower() or "ai" in c.claim.lower() for c in verification.verified_claims
        ):
            scores["technology"] = 100.0
        elif verification and any(
            "tech" in c.claim.lower() or "ai" in c.claim.lower() for c in verification.partially_verified_claims
        ):
            scores["technology"] = 60.0
        elif research and research.technology_signals and research.evidence:
            scores["technology"] = 30.0
        else:
            scores["technology"] = 0.0

        # 3. Timing / Urgency
        has_timing_contradiction = any("timing" in c.claim.lower() for c in contradicted_claims)
        if has_timing_contradiction:
            scores["timing"] = 0.0
        elif verification and any("timing" in c.claim.lower() for c in verification.verified_claims):
            scores["timing"] = 100.0
        elif verification and any("timing" in c.claim.lower() for c in verification.partially_verified_claims):
            scores["timing"] = 60.0
        elif research and any(e.source.published_at for e in research.evidence if e.source):
            scores["timing"] = 30.0
        else:
            scores["timing"] = 0.0

        # 4. Commercial relevance
        has_comm_contradiction = any(
            "commercial" in c.claim.lower() or "business" in c.claim.lower() for c in contradicted_claims
        )
        if has_comm_contradiction:
            scores["commercial"] = 0.0
        elif verification and any(
            "commercial" in c.claim.lower() or "business" in c.claim.lower() for c in verification.verified_claims
        ):
            scores["commercial"] = 100.0
        elif verification and any(
            "commercial" in c.claim.lower() or "business" in c.claim.lower()
            for c in verification.partially_verified_claims
        ):
            scores["commercial"] = 60.0
        elif research and research.business_signals and research.evidence:
            scores["commercial"] = 30.0
        else:
            scores["commercial"] = 0.0

        # 5. Prospect type
        has_pt_contradiction = any("prospect" in c.claim.lower() for c in contradicted_claims)
        if has_pt_contradiction:
            scores["prospect_type"] = 0.0
        elif qualification and qualification.prospect_type != ProspectType.UNKNOWN:
            if verification and verification.verification_status == VerificationStatus.VERIFIED:
                scores["prospect_type"] = 100.0
            elif verification and verification.verification_status == VerificationStatus.PARTIALLY_VERIFIED:
                scores["prospect_type"] = 60.0
            elif research and research.evidence:
                scores["prospect_type"] = 30.0
            else:
                scores["prospect_type"] = 0.0
        else:
            scores["prospect_type"] = 0.0

        # Normalization: All 5 categories are standard. Sum of weights = 1.0
        total_weight = sum(categories.values())
        weighted_sum = sum(scores[cat] * (categories[cat] / total_weight) for cat in categories)

        # Covered claims count (claims with coverage score >= 60)
        covered_count = sum(1 for s in scores.values() if s >= 60.0)
        total_count = len(categories)

        if scores["need"] >= 60.0:
            strengths.append("Verified or adequately supported need/problem evidence.")
        if scores["technology"] >= 60.0:
            strengths.append("Verified technology capability signals.")
        if scores["timing"] == 0.0 and not has_timing_contradiction:
            weaknesses.append("No timing or urgency evidence available.")

        return round(weighted_sum, 1), covered_count, total_count

    def _calculate_consistency(
        self,
        candidate: Candidate | None,
        research: ResearchResult | None,
        qualification: QualificationResult | None,
        verification: VerificationResult | None,
        scoring: ScoringResult | None,
        warnings: list[str],
        weaknesses: list[str],
    ) -> float:
        """Calculate consistency score (Weights: Prospect-type 25%, Need 30%, Timing 20%, Ver/Scoring 25%)."""
        # Dimensions & base weights
        weights = {
            "prospect_type": 0.25,
            "need": 0.30,
            "timing": 0.20,
            "ver_scoring": 0.25,
        }
        dim_scores: dict[str, float] = {}

        # 1. Prospect-type consistency (25%)
        # 100 consistent, 50 minor inconsistency, 0 material contradiction
        if qualification and qualification.prospect_type == ProspectType.UNKNOWN:
            dim_scores["prospect_type"] = 50.0
            weaknesses.append("Prospect-type consistency gap: qualification prospect type is UNKNOWN.")
        elif qualification and qualification.prospect_type == ProspectType.COMPETITOR_OR_VENDOR:
            if scoring and scoring.scores.priority in (Priority.HIGH, Priority.QUALIFIED):
                dim_scores["prospect_type"] = 0.0
                msg = "Prospect-type contradiction: candidate qualified as COMPETITOR_OR_VENDOR but scored as HIGH/QUALIFIED priority."
                warnings.append(msg)
                weaknesses.append(msg)
            else:
                dim_scores["prospect_type"] = 100.0
        else:
            dim_scores["prospect_type"] = 100.0

        # 2. Need/problem consistency (30%)
        has_need_contradiction = bool(
            verification and any("need" in c.claim.lower() for c in verification.contradicted_claims)
        )
        if has_need_contradiction:
            dim_scores["need"] = 0.0
            msg = "Need consistency contradiction: verified contradiction in candidate need."
            warnings.append(msg)
            weaknesses.append(msg)
        elif qualification and qualification.need_fit == FitLevel.LOW and scoring and scoring.scores.need_fit >= 70.0:
            dim_scores["need"] = 50.0
            weaknesses.append("Need consistency gap: qualification need fit was LOW while scoring need fit is high.")
        elif (not research or not research.problem_signals) and scoring and scoring.scores.need_fit >= 80.0:
            dim_scores["need"] = 50.0
            weaknesses.append("Need consistency gap: scoring need fit is high despite absent problem signals.")
        else:
            dim_scores["need"] = 100.0

        # 3. Timing consistency (20%)
        has_timing_contradiction = bool(
            verification and any("timing" in c.claim.lower() for c in verification.contradicted_claims)
        )
        if has_timing_contradiction:
            dim_scores["timing"] = 0.0
            msg = "Timing consistency contradiction: verified contradiction in timing signals."
            warnings.append(msg)
            weaknesses.append(msg)
        elif scoring and scoring.scores.timing >= 80.0 and (not research or not any(e.source.published_at for e in research.evidence if e.source)):
            dim_scores["timing"] = 50.0
            weaknesses.append("Timing consistency gap: high timing score without verified published dates.")
        else:
            dim_scores["timing"] = 100.0

        # 4. Verification/scoring consistency (25%)
        if scoring and scoring.verification_status == VerificationStatus.CONTRADICTED:
            if scoring.scores.priority in (Priority.HIGH, Priority.QUALIFIED):
                dim_scores["ver_scoring"] = 0.0
                msg = "Verification/scoring contradiction: candidate is CONTRADICTED but scored with active priority."
                warnings.append(msg)
                weaknesses.append(msg)
            else:
                dim_scores["ver_scoring"] = 100.0
        elif scoring and scoring.verification_status == VerificationStatus.UNVERIFIED and scoring.scores.confidence_score >= 80.0:
            dim_scores["ver_scoring"] = 50.0
            weaknesses.append("Verification/scoring gap: unverified evidence base paired with high confidence score.")
        else:
            dim_scores["ver_scoring"] = 100.0

        weighted_sum = sum(dim_scores[d] * weights[d] for d in weights)
        return round(weighted_sum, 1)

    def _calculate_actionability(
        self,
        candidate: Candidate | None,
        research: ResearchResult | None,
        qualification: QualificationResult | None,
        verification: VerificationResult | None,
        scoring: ScoringResult | None,
        operator_profile: OperatorProfile | None,
        strengths: list[str],
        weaknesses: list[str],
    ) -> float:
        """Calculate actionability score (100 strong, 50 partial, 0 absent across 6 components)."""
        # Weights:
        # Identifiable problem: 25%, Evidence supporting need: 20%, Clear potential fit: 20%,
        # Timing signal: 10%, Clear human next step: 15%, Company accessibility: 10%

        # 1. Identifiable problem/need (25%)
        if research and research.problem_signals:
            a_problem = 100.0 if len(research.problem_signals) >= 2 else 50.0
        elif qualification and qualification.reasons:
            a_problem = 50.0
        else:
            a_problem = 0.0

        # 2. Evidence supporting need (20%)
        if verification and verification.verification_status == VerificationStatus.CONTRADICTED:
            a_evidence = 0.0
        elif verification and verification.verification_status == VerificationStatus.VERIFIED:
            a_evidence = 100.0
        elif (research and research.evidence) or (verification and verification.partially_verified_claims):
            a_evidence = 50.0
        else:
            a_evidence = 0.0

        # 3. Clear potential fit (20%)
        if qualification and qualification.qualification_status == QualificationStatus.DISQUALIFIED:
            a_fit = 0.0
        elif operator_profile and operator_profile.capabilities and (research and research.problem_signals):
            a_fit = 100.0
        elif operator_profile and operator_profile.capabilities:
            a_fit = 50.0
        elif research and research.problem_signals:
            a_fit = 50.0
        else:
            a_fit = 0.0

        # 4. Timing signal (10%)
        has_dates = bool(research and any(e.source.published_at for e in research.evidence if e.source))
        if qualification and qualification.timing == FitLevel.HIGH and has_dates:
            a_timing = 100.0
        elif has_dates or (qualification and qualification.timing in (FitLevel.HIGH, FitLevel.MEDIUM)):
            a_timing = 50.0
        else:
            a_timing = 0.0

        # 5. Clear human next step (15%)
        if verification and verification.verification_status == VerificationStatus.CONTRADICTED:
            a_step = 100.0  # Clear step: resolve contradiction
        elif qualification and qualification.qualification_status == QualificationStatus.DISQUALIFIED:
            a_step = 100.0  # Clear step: discard/no action
        elif scoring and scoring.scores.priority in (Priority.HIGH, Priority.QUALIFIED, Priority.WATCHLIST, Priority.DISCARD):
            a_step = 100.0
        else:
            a_step = 50.0

        # 6. Company accessibility (10%)
        has_site = bool((research and research.website) or (candidate and candidate.website))
        has_dm = bool(research and research.people and any(p.name or p.role for p in research.people))
        if has_site and has_dm:
            a_access = 100.0
            strengths.append("Directly accessible: verified website and identified decision maker.")
        elif has_site:
            a_access = 50.0
        else:
            a_access = 0.0
            weaknesses.append("Limited company accessibility: missing verified website.")

        score = (
            a_problem * 0.25
            + a_evidence * 0.20
            + a_fit * 0.20
            + a_timing * 0.10
            + a_step * 0.15
            + a_access * 0.10
        )
        return round(score, 1)

    def _synthesize_signals(
        self,
        research: ResearchResult | None,
        verification: VerificationResult | None,
        strengths: list[str],
    ) -> list[str]:
        """Synthesize evidence-backed signal categories: HIRING, PRODUCT, TECHNOLOGY, BUSINESS, GROWTH, PROBLEM."""
        # Converging signals require at least 2 distinct evidence-backed categories with no material contradiction
        if not research or not research.evidence:
            return []

        if verification and (
            verification.verification_status == VerificationStatus.CONTRADICTED
            or bool(verification.contradicted_claims)
        ):
            return []

        detected_categories: set[str] = set()

        # Check evidence and signals
        for ev in research.evidence:
            text = (ev.claim + " " + ev.evidence_summary).lower()
            if any(k in text for k in ("hiring", "job", "recruiting", "opening", "career", "role")):
                detected_categories.add("HIRING")
            if any(k in text for k in ("product", "release", "launch", "feature", "platform", "version")):
                detected_categories.add("PRODUCT")
            if any(k in text for k in ("ai", "technology", "stack", "framework", "model", "pipeline", "infrastructure")):
                detected_categories.add("TECHNOLOGY")
            if any(k in text for k in ("partnership", "customer", "contract", "client", "market", "business")):
                detected_categories.add("BUSINESS")
            if any(k in text for k in ("growth", "scaling", "expansion", "series a", "series b", "funding", "revenue")):
                detected_categories.add("GROWTH")
            if any(k in text for k in ("problem", "challenge", "bottleneck", "pain point", "issue", "struggling")):
                detected_categories.add("PROBLEM")

        # Also inspect explicit signals on research
        if research.technology_signals:
            detected_categories.add("TECHNOLOGY")
        for sig in research.business_signals:
            sig_lower = sig.lower()
            if any(k in sig_lower for k in ("hiring", "job", "recruiting", "opening", "career", "role")):
                detected_categories.add("HIRING")
            elif any(k in sig_lower for k in ("growth", "scaling", "expansion", "series a", "series b", "funding", "revenue")):
                detected_categories.add("GROWTH")
            else:
                detected_categories.add("BUSINESS")
        if research.problem_signals:
            detected_categories.add("PROBLEM")

        if len(detected_categories) < 2:
            return []

        # Deterministic sorting of categories
        sorted_cats = sorted(detected_categories)
        synthesis_statement = (
            f"Converging signals across {', '.join(sorted_cats)} indicate active, observable business activity."
        )
        strengths.append(f"Multiple converging evidence categories detected: {', '.join(sorted_cats)}.")
        return [synthesis_statement]

    def _build_narrative(
        self,
        candidate: Candidate | None,
        research: ResearchResult | None,
        qualification: QualificationResult | None,
        verification: VerificationResult | None,
        scoring: ScoringResult | None,
        operator_profile: OperatorProfile | None,
    ) -> OpportunityNarrative:
        """Construct deterministic, evidence-derived Opportunity Narrative without LLM calls."""
        # 1. why_company
        company_name = (
            (candidate and candidate.company_name)
            or (scoring and scoring.company_name)
            or "Company"
        )
        if candidate and candidate.discovery_reason:
            why_company = f"{company_name} surfaced via {candidate.discovery_strategy}: {candidate.discovery_reason}."
        elif candidate:
            why_company = f"{company_name} surfaced via strategy {candidate.discovery_strategy}."
        else:
            why_company = f"{company_name} identified through opportunity research."

        # 2. why_now
        timing_signals: list[str] = []
        if research and research.evidence:
            for ev in research.evidence:
                if ev.source and ev.source.published_at:
                    dt_str = ev.source.published_at.strftime("%Y-%m-%d")
                    timing_signals.append(f"evidence dated {dt_str} ({ev.claim})")
        if timing_signals:
            why_now = f"Active timing signals observed: {'; '.join(timing_signals[:2])}."
        elif qualification and qualification.timing == FitLevel.HIGH:
            why_now = "Recent operational timing signals indicated by high qualification urgency."
        else:
            why_now = "No recent timing signal identified."

        # 3. identified_problem
        if research and research.problem_signals:
            identified_problem = "; ".join(research.problem_signals)
        elif qualification and qualification.reasons:
            identified_problem = "; ".join(qualification.reasons)
        else:
            identified_problem = None

        # 4. why_fit
        if not operator_profile or not operator_profile.capabilities:
            why_fit = "Operator profile not configured; fit cannot be determined without defined capabilities."
        elif identified_problem:
            caps = ", ".join(operator_profile.capabilities)
            why_fit = f"Operator capabilities ({caps}) align with identified challenge ({identified_problem})."
        else:
            caps = ", ".join(operator_profile.capabilities)
            why_fit = f"Operator capabilities ({caps}) match general sector opportunities."

        # 5. evidence_summary
        ev_items: list[str] = []
        if verification and verification.verified_claims:
            claims_str = "; ".join(c.claim for c in verification.verified_claims)
            ev_items.append(f"Verified claims: {claims_str}")
        elif research and research.evidence:
            claims_str = "; ".join(e.claim for e in research.evidence[:3])
            ev_items.append(f"Observed evidence claims: {claims_str}")
        else:
            ev_items.append("No structured evidence records available.")
        evidence_summary = ". ".join(ev_items) + "."

        # 6. next_human_step
        if verification and (
            verification.verification_status == VerificationStatus.CONTRADICTED
            or bool(verification.contradicted_claims)
        ):
            next_human_step = "Resolve the conflicting evidence before considering this opportunity."
        elif qualification and qualification.qualification_status == QualificationStatus.DISQUALIFIED:
            next_human_step = "No action recommended; candidate was disqualified during qualification."
        elif scoring and scoring.scores.priority == Priority.DISCARD:
            next_human_step = "No action recommended; candidate did not meet opportunity thresholds."
        elif scoring and scoring.scores.confidence_score < 50.0:
            next_human_step = "Review the evidence gaps before taking any action."
        elif scoring and scoring.scores.priority == Priority.HIGH:
            next_human_step = "Operator should conduct manual outreach preparation and validate identified need."
        elif scoring and scoring.scores.priority == Priority.QUALIFIED:
            next_human_step = "Operator should review supporting evidence and determine feasibility of human follow-up."
        elif scoring and scoring.scores.priority == Priority.WATCHLIST:
            next_human_step = "Monitor the identified signal and reassess if stronger evidence of an active need appears."
        else:
            next_human_step = "Review the supporting evidence and determine next human action."

        return OpportunityNarrative(
            why_company=why_company,
            why_now=why_now,
            identified_problem=identified_problem,
            why_fit=why_fit,
            evidence_summary=evidence_summary,
            next_human_step=next_human_step,
        )
