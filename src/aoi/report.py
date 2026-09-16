from datetime import UTC, datetime

from .schemas.common import Priority, RunStatus, VerificationStatus
from .schemas.discovery import Candidate, DiscoveryEvaluation
from .schemas.intelligence import IntelligenceQualityResult, IntelligenceQualityStatus
from .schemas.objective import BusinessObjective, OperatorProfile
from .schemas.qualification import ProspectType, QualificationResult, QualificationStatus
from .schemas.report import (
    AOIReport,
    ReportEvidence,
    ReportOpportunity,
    ReportStatistics,
    ReportSummary,
)
from .schemas.research import ResearchEvaluation, ResearchResult
from .schemas.scoring import ScoringResult
from .schemas.verification import VerificationResult

PRIORITY_SORT_ORDER: dict[Priority, int] = {
    Priority.HIGH: 0,
    Priority.QUALIFIED: 1,
    Priority.WATCHLIST: 2,
    Priority.DISCARD: 3,
}

HIGH_CONFIDENCE_THRESHOLD: float = 80.0


def deduplicate_warnings(warnings: list[str]) -> list[str]:
    """Deterministically deduplicate warnings while preserving original ordering."""
    seen: set[str] = set()
    deduped: list[str] = []
    for w in warnings:
        clean = w.strip() if isinstance(w, str) else str(w)
        if clean and clean not in seen:
            seen.add(clean)
            deduped.append(clean)
    return deduped


class ReportBuilder:
    """Transforms existing AOI pipeline state into a validated AOIReport.

    Deterministic V1 presentation layer. Does not call LLMs, external APIs,
    nor recalculate scores, priorities, or statuses.
    """

    def build(
        self,
        state=None,
        *,
        run_id: str | None = None,
        objective: dict | BusinessObjective | None = None,
        status: RunStatus | None = None,
        candidates: list[Candidate] | None = None,
        research_results: list[ResearchResult] | None = None,
        qualification_results: list[QualificationResult] | None = None,
        verification_results: list[VerificationResult] | None = None,
        scoring_results: list[ScoringResult] | None = None,
        quality_results: list[IntelligenceQualityResult] | None = None,
        discovery_evaluation: DiscoveryEvaluation | None = None,
        research_evaluation: ResearchEvaluation | None = None,
        operator_profile: OperatorProfile | None = None,
        warnings: list[str] | None = None,
        generated_at: datetime | None = None,
    ) -> AOIReport:
        """Build a deterministic AOIReport from state or explicit components."""
        # Extract from state if provided
        if state is not None:
            run_id = run_id or getattr(state, "run_id", "AOI-RUN")
            if objective is None:
                state_input = getattr(state, "input", None)
                objective = getattr(state_input, "objective", {}) if state_input else {}
            if operator_profile is None:
                state_input = getattr(state, "input", None)
                operator_profile = getattr(state_input, "operator_profile", None) if state_input else None
            status = status or getattr(state, "status", RunStatus.COMPLETED)
            candidates = candidates if candidates is not None else list(getattr(state, "candidates", []))
            research_results = (
                research_results
                if research_results is not None
                else list(getattr(state, "research_results", []))
            )
            qualification_results = (
                qualification_results
                if qualification_results is not None
                else list(getattr(state, "qualification_results", []))
            )
            verification_results = (
                verification_results
                if verification_results is not None
                else list(getattr(state, "verification_results", []))
            )
            scoring_results = (
                scoring_results
                if scoring_results is not None
                else list(getattr(state, "scoring_results", []))
            )
            quality_results = (
                quality_results
                if quality_results is not None
                else list(getattr(state, "quality_results", []))
            )
            discovery_evaluation = discovery_evaluation or getattr(state, "discovery_evaluation", None)
            research_evaluation = research_evaluation or getattr(state, "research_evaluation", None)
            warnings = warnings if warnings is not None else list(getattr(state, "warnings", []))

        # Defaults for missing metadata
        run_id = run_id or "AOI-RUN"
        status = status or RunStatus.COMPLETED
        generated_at = generated_at or datetime.now(UTC)
        candidates = candidates or []
        research_results = research_results or []
        qualification_results = qualification_results or []
        verification_results = verification_results or []
        scoring_results = scoring_results or []
        quality_results = quality_results or []
        warnings = warnings or []

        # Convert objective to dict
        if hasattr(objective, "model_dump"):
            objective_dict = objective.model_dump()
        elif isinstance(objective, dict):
            objective_dict = objective
        elif objective is not None:
            objective_dict = {"description": str(objective)}
        else:
            objective_dict = {}

        # Index upstream models by candidate_id
        candidate_by_id = {c.candidate_id: c for c in candidates}
        research_by_id = {r.candidate_id: r for r in research_results}
        qualification_by_id = {q.candidate_id: q for q in qualification_results}
        verification_by_id = {v.candidate_id: v for v in verification_results}
        quality_by_id = {q.candidate_id: q for q in quality_results}

        # Deterministic sorting of scoring results
        # 1. Priority: HIGH, QUALIFIED, WATCHLIST, DISCARD
        # 2. Opportunity Score descending
        # 3. Confidence Score descending
        # 4. Company name ascending
        # 5. candidate_id ascending
        def sort_key(s: ScoringResult):
            priority_idx = PRIORITY_SORT_ORDER.get(s.scores.priority, 99)
            return (
                priority_idx,
                -s.scores.opportunity_score,
                -s.scores.confidence_score,
                s.company_name,
                s.candidate_id,
            )

        sorted_scored = sorted(scoring_results, key=sort_key)

        opportunities: list[ReportOpportunity] = []
        watchlist: list[ReportOpportunity] = []

        # Ranks are global across non-discarded entries (HIGH, QUALIFIED, WATCHLIST)
        current_rank = 1
        for s in sorted_scored:
            if s.scores.priority == Priority.DISCARD:
                continue

            cand = candidate_by_id.get(s.candidate_id)
            research = research_by_id.get(s.candidate_id)
            qual = qualification_by_id.get(s.candidate_id)
            ver = verification_by_id.get(s.candidate_id)
            qual_res = quality_by_id.get(s.candidate_id)

            report_opp = self._build_opportunity(
                rank=current_rank,
                scoring=s,
                candidate=cand,
                research=research,
                qualification=qual,
                verification=ver,
                operator_profile=operator_profile,
                quality_result=qual_res,
            )
            current_rank += 1

            if s.scores.priority in (Priority.HIGH, Priority.QUALIFIED):
                opportunities.append(report_opp)
            elif s.scores.priority == Priority.WATCHLIST:
                watchlist.append(report_opp)

        # Statistics computation
        high_count = sum(1 for s in scoring_results if s.scores.priority == Priority.HIGH)
        qualified_count = sum(1 for s in scoring_results if s.scores.priority == Priority.QUALIFIED)
        watchlist_count = sum(1 for s in scoring_results if s.scores.priority == Priority.WATCHLIST)
        
        # Count discard from scoring, plus any unscored candidates disqualified in qualification
        scored_ids = {s.candidate_id for s in scoring_results}
        unscored_disqualified = [
            q for q in qualification_results
            if q.qualification_status == QualificationStatus.DISQUALIFIED and q.candidate_id not in scored_ids
        ]
        discard_count = sum(1 for s in scoring_results if s.scores.priority == Priority.DISCARD) + len(unscored_disqualified)

        queries_exec = sum(r.queries_executed for r in research_results)
        queries_sav = sum(r.queries_saved for r in research_results)
        total_q = queries_exec + queries_sav
        cost_savings = round((queries_sav / max(total_q, 1)) * 100.0, 1)
        multi_sig_cands = sum(1 for c in candidates if len(set(c.signal_categories)) >= 2)

        statistics = ReportStatistics(
            candidates_discovered=len(candidates),
            candidates_researched=len(research_results),
            candidates_qualified=len(qualification_results),
            candidates_verified=len(verification_results),
            opportunities_scored=len(scoring_results),
            high_priority=high_count,
            qualified_priority=qualified_count,
            watchlist_priority=watchlist_count,
            discard_priority=discard_count,
            discovery_evaluation=discovery_evaluation,
            research_evaluation=research_evaluation,
            queries_executed=queries_exec,
            queries_saved=queries_sav,
            cost_savings_percentage=cost_savings,
            multi_signal_candidates=multi_sig_cands,
        )

        # Aggregate warnings deterministically
        aggregated_warnings: list[str] = []
        aggregated_warnings.extend(warnings)
        for r in research_results:
            aggregated_warnings.extend(r.warnings)
        for q in qualification_results:
            aggregated_warnings.extend(q.warnings)
        for v in verification_results:
            aggregated_warnings.extend(v.verification_warnings)
        for s in scoring_results:
            aggregated_warnings.extend(s.scoring_warnings)
        for q in quality_results:
            aggregated_warnings.extend(q.quality_warnings)

        deduped_warnings = deduplicate_warnings(aggregated_warnings)

        # High confidence count
        high_confidence_count = sum(
            1 for s in scoring_results if s.scores.confidence_score >= HIGH_CONFIDENCE_THRESHOLD
        )
        top_opportunity_count = len(opportunities)

        # Deterministic summary
        if top_opportunity_count > 0:
            cand_len = len(candidates)
            cand_phrase = f" across {cand_len} discovered candidate{'s' if cand_len != 1 else ''}" if cand_len > 0 else ""
            headline = f"Identified {top_opportunity_count} actionable opportunit{'ies' if top_opportunity_count != 1 else 'y'}{cand_phrase}."
            overview = (
                f"Pipeline evaluated {len(scoring_results)} opportunities: "
                f"{high_count} High Priority, {qualified_count} Qualified, "
                f"{watchlist_count} Watchlist, {discard_count} Discarded. "
                f"{high_confidence_count} exhibited high evidence confidence."
            )
        else:
            headline = "No actionable opportunities identified."
            overview = (
                f"The pipeline evaluated {len(scoring_results)} opportunities, "
                "but none met the threshold for active outreach."
            )

        summary = ReportSummary(
            headline=headline,
            overview=overview,
            top_opportunity_count=top_opportunity_count,
            high_confidence_count=high_confidence_count,
            warning_count=len(deduped_warnings),
        )

        return AOIReport(
            run_id=run_id,
            objective=objective_dict,
            run_status=status,
            generated_at=generated_at,
            summary=summary,
            statistics=statistics,
            opportunities=opportunities,
            watchlist=watchlist,
            discarded_count=discard_count,
            warnings=deduped_warnings,
        )

    def _build_opportunity(
        self,
        rank: int,
        scoring: ScoringResult,
        candidate: Candidate | None,
        research: ResearchResult | None,
        qualification: QualificationResult | None,
        verification: VerificationResult | None,
        operator_profile: OperatorProfile | None = None,
        quality_result: IntelligenceQualityResult | None = None,
    ) -> ReportOpportunity:
        """Construct a validated ReportOpportunity without modifying upstream values."""
        # Website preservation
        website: str | None = None
        if research and research.website:
            website = str(research.website)
        elif candidate and candidate.website:
            website = str(candidate.website)

        # Prospect type & Qualification status
        prospect_type = qualification.prospect_type if qualification else ProspectType.UNKNOWN
        qualification_status = (
            qualification.qualification_status if qualification else QualificationStatus.INSUFFICIENT_EVIDENCE
        )

        # Problem signal extraction
        problem: str | None = None
        if research and research.problem_signals:
            problem = "; ".join(research.problem_signals)
        elif qualification and qualification.reasons:
            problem = "; ".join(qualification.reasons)

        # Potential solution: describe potential service direction only when supported by problem/capability
        potential_solution: str | None = None
        if problem:
            if operator_profile and operator_profile.capabilities:
                caps = ", ".join(operator_profile.capabilities)
                potential_solution = f"Potential service direction: apply capabilities ({caps}) to address {problem}."
            else:
                potential_solution = f"Potential service direction: address identified technical need ({problem})."

        # Deterministic summary
        summary_parts: list[str] = []
        pt_display = prospect_type.value.replace("_", " ").lower()
        summary_parts.append(f"{scoring.company_name} identified as {pt_display}")
        if research and research.problem_signals:
            summary_parts.append(f"observed signals: {'; '.join(research.problem_signals)}")
        elif qualification and qualification.reasons:
            summary_parts.append(f"qualification notes: {'; '.join(qualification.reasons)}")
        if verification and verification.verified_claims:
            v_claims = "; ".join(c.claim for c in verification.verified_claims)
            summary_parts.append(f"verified: {v_claims}")
        summary_parts.append(
            f"scored priority {scoring.scores.priority.value} "
            f"(opportunity: {scoring.scores.opportunity_score:.1f}, confidence: {scoring.scores.confidence_score:.1f})"
        )
        opportunity_summary = ". ".join(summary_parts) + "."

        # Convert evidence preserving exact fields
        report_evidence: list[ReportEvidence] = []
        if research and research.evidence:
            for ev in research.evidence:
                report_evidence.append(
                    ReportEvidence(
                        evidence_id=ev.id,
                        claim=ev.claim,
                        source=str(ev.source.url),
                        source_type=ev.source.source_type,
                        published_at=ev.source.published_at,
                        accessed_at=ev.source.accessed_at,
                        evidence_summary=ev.evidence_summary,
                        reliability=float(ev.source.reliability),
                        recency=float(ev.recency_score),
                        verification_status=str(
                            ev.verification_status.value
                            if hasattr(ev.verification_status, "value")
                            else ev.verification_status
                        ),
                    )
                )

        # Candidate warnings aggregation
        opp_warnings: list[str] = []
        if research:
            opp_warnings.extend(research.warnings)
        if qualification:
            opp_warnings.extend(qualification.warnings)
        if verification:
            opp_warnings.extend(verification.verification_warnings)
        opp_warnings.extend(scoring.scoring_warnings)
        if quality_result:
            opp_warnings.extend(quality_result.quality_warnings)
        deduped_opp_warnings = deduplicate_warnings(opp_warnings)

        # Recommended human action
        recommended_action = self._determine_recommended_action(
            scoring=scoring,
            qualification=qualification,
            verification=verification,
        )

        quality = quality_result.quality if quality_result else None
        narrative = quality_result.narrative if quality_result else None
        strengths = list(quality_result.strengths) if quality_result else []
        weaknesses = list(quality_result.weaknesses) if quality_result else []
        converging_signals = list(quality_result.converging_signals) if quality_result else []
        quality_status = (
            IntelligenceQualityStatus.from_score(quality.quality_score)
            if quality
            else None
        )

        return ReportOpportunity(
            rank=rank,
            candidate_id=scoring.candidate_id,
            company_name=scoring.company_name,
            website=website,
            opportunity_summary=opportunity_summary,
            problem=problem,
            potential_solution=potential_solution,
            prospect_type=prospect_type,
            qualification_status=qualification_status,
            verification_status=scoring.verification_status,
            opportunity_score=scoring.scores.opportunity_score,
            confidence_score=scoring.scores.confidence_score,
            priority=scoring.scores.priority,
            score_breakdown=scoring.scores,
            evidence=report_evidence,
            scoring_reasons=list(scoring.scoring_reasons),
            warnings=deduped_opp_warnings,
            recommended_action=recommended_action,
            quality=quality,
            narrative=narrative,
            strengths=strengths,
            weaknesses=weaknesses,
            converging_signals=converging_signals,
            quality_status=quality_status,
        )

    def _determine_recommended_action(
        self,
        scoring: ScoringResult,
        qualification: QualificationResult | None,
        verification: VerificationResult | None,
    ) -> str:
        """Determine human-controlled recommended action based on verified upstream state."""
        # 1. Contradicted verification status
        if scoring.verification_status == VerificationStatus.CONTRADICTED or (
            verification and verification.verification_status == VerificationStatus.CONTRADICTED
        ):
            return "Resolve the conflicting evidence before considering this opportunity."

        # 2. Disqualified candidate
        if qualification and qualification.qualification_status == QualificationStatus.DISQUALIFIED:
            return "No action recommended; candidate was disqualified during qualification."

        # 3. Discard priority
        if scoring.scores.priority == Priority.DISCARD:
            return "No action recommended; candidate did not meet opportunity thresholds."

        # 4. Low confidence
        if scoring.scores.confidence_score < 50.0:
            return "Review the evidence gaps before taking any action."

        # 5. Priority HIGH
        if scoring.scores.priority == Priority.HIGH:
            return "Review this opportunity first and validate the identified need before initiating human outreach."

        # 6. Priority QUALIFIED
        if scoring.scores.priority == Priority.QUALIFIED:
            return "Review the supporting evidence and determine whether this opportunity merits human follow-up."

        # 7. Priority WATCHLIST
        if scoring.scores.priority == Priority.WATCHLIST:
            return "Monitor the identified signal and reassess if stronger evidence of an active need appears."

        return "Review the supporting evidence and determine next human action."
