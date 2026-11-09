"""
Project SHIFT: Model Diagnostics & Failure Mode Taxonomy.
Component 7 of SHIFT v0.3 Scientific Benchmark Infrastructure.

Identifies model weaknesses and categorizes semantic parsing mistakes into
five diagnostic failure modes:
1. LexicalInductionFailure: Misaligned root entity/action morphemes.
2. MorphosyntacticFailure: Misparsed tense or polarity affixes.
3. DiscourseTrackingFailure: Misresolved coreference, ellipsis, or correction.
4. DisentanglementFailure: Correct semantics but failed causal drift attribution.
5. CounterfactualContextBias: Rigid output invariant to context perturbation.

Generates actionable pedagogical feedback for contestants and comprehensive
diagnostic profiles for coaches and the IOAI Scientific Committee.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, Counter

from shift.ontology import EventFrame, DriftSource
from shift.episodes import Episode


class FailureMode(str, Enum):
    """The five standard diagnostic failure modes in Project SHIFT."""
    LEXICAL_INDUCTION = "LexicalInductionFailure"
    MORPHOSYNTACTIC = "MorphosyntacticFailure"
    DISCOURSE_TRACKING = "DiscourseTrackingFailure"
    DISENTANGLEMENT = "DisentanglementFailure"
    COUNTERFACTUAL_BIAS = "CounterfactualContextBias"


@dataclass
class EpisodeDiagnostic:
    """Detailed error diagnosis for an individual test episode."""
    episode_id: str
    difficulty_level: int
    phenomenon: str
    is_exact_match: bool
    failure_modes: List[FailureMode] = field(default_factory=list)
    explanation: str = ""
    slot_errors: List[str] = field(default_factory=list)
    predicted_frame: Optional[Dict[str, Any]] = None
    target_frame: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "difficulty_level": self.difficulty_level,
            "phenomenon": self.phenomenon,
            "is_exact_match": self.is_exact_match,
            "failure_modes": [m.value for m in self.failure_modes],
            "slot_errors": self.slot_errors,
            "explanation": self.explanation,
            "predicted_frame": self.predicted_frame,
            "target_frame": self.target_frame
        }


@dataclass
class DiagnosticReport:
    """Comprehensive diagnostic profile of a model across a test suite."""
    total_episodes: int
    total_errors: int
    exact_match_accuracy: float
    failure_counts: Dict[str, int]
    failure_percentages: Dict[str, float]
    slot_error_counts: Dict[str, int]
    phenomenon_error_rates: Dict[str, float]
    level_error_rates: Dict[str, float]
    episode_diagnostics: List[EpisodeDiagnostic]
    recommendations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "total_errors": self.total_errors,
            "exact_match_accuracy": self.exact_match_accuracy,
            "failure_counts": self.failure_counts,
            "failure_percentages": self.failure_percentages,
            "slot_error_counts": self.slot_error_counts,
            "phenomenon_error_rates": self.phenomenon_error_rates,
            "level_error_rates": self.level_error_rates,
            "recommendations": self.recommendations,
            "sample_diagnostics": [d.to_dict() for d in self.episode_diagnostics[:10]]
        }

    def format_text_summary(self) -> str:
        """Formats diagnostic report as human-readable string."""
        lines = [
            "=" * 65,
            " MODEL DIAGNOSTIC PROFILE & ERROR TAXONOMY",
            "=" * 65,
            f"Total Evaluated: {self.total_episodes} | Failed Episodes: {self.total_errors}",
            f"Overall Exact Match: {self.exact_match_accuracy:.2f}%\n",
            "Root-Cause Failure Mode Breakdown:",
            "  " + "-" * 60
        ]

        for mode, count in sorted(self.failure_counts.items(), key=lambda x: -x[1]):
            pct = self.failure_percentages.get(mode, 0.0)
            lines.append(f"  {mode:<28} : {count:4d} instances ({pct:5.1f}% of errors)")

        lines.extend([
            "\nSlot Error Distribution:",
            "  " + "-" * 60
        ])
        for slot, count in sorted(self.slot_error_counts.items(), key=lambda x: -x[1]):
            lines.append(f"  {slot:<16} : {count:4d} errors")

        lines.extend([
            "\nVulnerability by Dialogue Phenomenon:",
            "  " + "-" * 60
        ])
        for phen, rate in sorted(self.phenomenon_error_rates.items(), key=lambda x: -x[1]):
            lines.append(f"  {phen:<16} : {rate:5.1f}% error rate")

        lines.extend([
            "\nVulnerability by Difficulty Level:",
            "  " + "-" * 60
        ])
        for lvl, rate in sorted(self.level_error_rates.items()):
            lines.append(f"  {lvl:<16} : {rate:5.1f}% error rate")

        lines.extend([
            "\nPedagogical Recommendations for Contestants & Coaches:",
            "  " + "-" * 60
        ])
        for rec in self.recommendations:
            lines.append(f"  * {rec}")

        lines.append("=" * 65 + "\n")
        return "\n".join(lines)


class ModelDiagnostics:
    """Diagnoses model mistakes and produces structured diagnostic reports."""

    @classmethod
    def diagnose_episode(
        cls,
        episode: Episode,
        prediction: Optional[EventFrame],
        counterpart_pred: Optional[EventFrame] = None,
        counterpart_ep: Optional[Episode] = None
    ) -> EpisodeDiagnostic:
        """Diagnoses a single episode against ground truth."""
        gt = episode.query_label
        if gt is None:
            return EpisodeDiagnostic(
                episode_id=episode.episode_id,
                difficulty_level=episode.difficulty_level,
                phenomenon=episode.query.phenomenon,
                is_exact_match=False,
                explanation="No ground truth label available."
            )

        if prediction is None:
            return EpisodeDiagnostic(
                episode_id=episode.episode_id,
                difficulty_level=episode.difficulty_level,
                phenomenon=episode.query.phenomenon,
                is_exact_match=False,
                failure_modes=[FailureMode.LEXICAL_INDUCTION],
                explanation="Missing prediction (Null output).",
                target_frame=gt.to_dict()
            )

        slot_errors = []
        for slot in ["agent", "action", "object", "tense", "polarity"]:
            if getattr(prediction, slot) != getattr(gt, slot):
                slot_errors.append(slot)

        ref_error = (prediction.resolved_references != gt.resolved_references)
        disentangle_error = False
        if gt.drift_attribution and prediction.drift_attribution != gt.drift_attribution:
            disentangle_error = True

        failure_modes: List[FailureMode] = []
        explanations: List[str] = []

        # 1. Lexical Induction Failure
        lexical_slots = [s for s in slot_errors if s in ["agent", "action", "object"]]
        if lexical_slots:
            failure_modes.append(FailureMode.LEXICAL_INDUCTION)
            explanations.append(
                f"Lexical induction failure on {', '.join(lexical_slots)}: "
                f"expected ({', '.join(getattr(gt, s) for s in lexical_slots)}) but predicted "
                f"({', '.join(getattr(prediction, s) for s in lexical_slots)})."
            )

        # 2. Morphosyntactic Failure
        morpho_slots = [s for s in slot_errors if s in ["tense", "polarity"]]
        if morpho_slots:
            failure_modes.append(FailureMode.MORPHOSYNTACTIC)
            explanations.append(
                f"Morphosyntactic inflection failure on {', '.join(morpho_slots)}: "
                f"failed to parse tense affix or polarity marker."
            )

        # 3. Discourse Tracking Failure
        if ref_error or (episode.query.phenomenon in ["coreference", "ellipsis", "correction"] and "agent" in slot_errors):
            failure_modes.append(FailureMode.DISCOURSE_TRACKING)
            explanations.append(
                f"Discourse tracking failure in '{episode.query.phenomenon}': "
                f"failed to resolve antecedent across conversational turns."
            )

        # 4. Disentanglement Failure
        # If event semantics are correct (or mostly correct) but drift attribution is wrong
        if len(slot_errors) == 0 and not ref_error and disentangle_error:
            failure_modes.append(FailureMode.DISENTANGLEMENT)
            explanations.append(
                f"Disentanglement failure: correctly parsed event frame but misattributed "
                f"causal drift factors (got {prediction.drift_attribution}, expected {gt.drift_attribution})."
            )

        # 5. Counterfactual Context Bias
        if counterpart_pred and counterpart_ep and counterpart_ep.query_label:
            # If the two counterpart ground truths differ, but prediction is identical
            if counterpart_ep.query_label.to_dict() != gt.to_dict():
                if prediction.to_dict() == counterpart_pred.to_dict():
                    failure_modes.append(FailureMode.COUNTERFACTUAL_BIAS)
                    explanations.append(
                        "Counterfactual context bias: model produced identical outputs across "
                        "contextually divergent counterfactual branches."
                    )

        is_exact = (len(slot_errors) == 0 and not ref_error and not disentangle_error)

        return EpisodeDiagnostic(
            episode_id=episode.episode_id,
            difficulty_level=episode.difficulty_level,
            phenomenon=episode.query.phenomenon,
            is_exact_match=is_exact,
            failure_modes=failure_modes,
            slot_errors=slot_errors,
            explanation=" ".join(explanations) if explanations else "All event slots and attributions matched perfectly.",
            predicted_frame=prediction.to_dict(),
            target_frame=gt.to_dict()
        )

    @classmethod
    def diagnose_all(
        cls,
        episodes: List[Episode],
        predictions: Dict[str, EventFrame]
    ) -> DiagnosticReport:
        """Diagnoses an entire collection of test episodes."""
        total = len(episodes)
        if total == 0:
            raise ValueError("Cannot diagnose an empty list of episodes.")

        # Map counterfactual pairs
        cf_pairs: Dict[str, List[Episode]] = defaultdict(list)
        episodes_by_id = {ep.episode_id: ep for ep in episodes}
        for ep in episodes:
            if ep.counterfactual_pair_id:
                cf_pairs[ep.counterfactual_pair_id].append(ep)

        diagnostics: List[EpisodeDiagnostic] = []
        failure_counts: Dict[str, int] = defaultdict(int)
        slot_error_counts: Dict[str, int] = defaultdict(int)

        phenomenon_totals = defaultdict(int)
        phenomenon_fails = defaultdict(int)

        level_totals = defaultdict(int)
        level_fails = defaultdict(int)

        exact_count = 0
        total_errors = 0

        for ep in episodes:
            pred = predictions.get(ep.episode_id)

            # Check counterpart
            cp_pred = None
            cp_ep = None
            if ep.counterfactual_pair_id and len(cf_pairs[ep.counterfactual_pair_id]) == 2:
                pair = cf_pairs[ep.counterfactual_pair_id]
                cp_ep = pair[1] if pair[0].episode_id == ep.episode_id else pair[0]
                cp_pred = predictions.get(cp_ep.episode_id)

            diag = cls.diagnose_episode(ep, pred, counterpart_pred=cp_pred, counterpart_ep=cp_ep)
            diagnostics.append(diag)

            phenomenon_totals[ep.query.phenomenon] += 1
            level_totals[f"Level_{ep.difficulty_level}"] += 1

            if diag.is_exact_match:
                exact_count += 1
            else:
                total_errors += 1
                phenomenon_fails[ep.query.phenomenon] += 1
                level_fails[f"Level_{ep.difficulty_level}"] += 1

                for mode in diag.failure_modes:
                    failure_counts[mode.value] += 1
                for slot in diag.slot_errors:
                    slot_error_counts[slot] += 1

        # Error percentages relative to total errors
        failure_percentages = {
            m: round(cnt / total_errors * 100.0, 2) if total_errors else 0.0
            for m, cnt in failure_counts.items()
        }

        phenomenon_error_rates = {
            p: round(phenomenon_fails[p] / phenomenon_totals[p] * 100.0, 2) if phenomenon_totals[p] else 0.0
            for p in sorted(phenomenon_totals.keys())
        }

        level_error_rates = {
            lvl: round(level_fails[lvl] / level_totals[lvl] * 100.0, 2) if level_totals[lvl] else 0.0
            for lvl in sorted(level_totals.keys())
        }

        # Pedagogical recommendations generator
        recommendations: List[str] = []
        if failure_counts.get(FailureMode.LEXICAL_INDUCTION.value, 0) > total_errors * 0.3:
            recommendations.append(
                "High Lexical Induction Failure: Your model is failing to bind novel root morphemes from the support set. "
                "Ensure your induction algorithm aligns word order constituents (SOV/SVO/VSO) before extracting mappings."
            )
        if failure_counts.get(FailureMode.MORPHOSYNTACTIC.value, 0) > total_errors * 0.2:
            recommendations.append(
                "Significant Morphosyntactic Failure: Check your affix segmentation module. Historical time periods introduce "
                "phonological lenitions and suffix overrides for tense (-il, -en, -or) and negation prefixes (no-)."
            )
        if failure_counts.get(FailureMode.DISCOURSE_TRACKING.value, 0) > total_errors * 0.2:
            recommendations.append(
                "Discourse Tracking Vulnerability: The model is struggling with coreference and conversational repair. "
                "Track speaker discourse state across turns and update salience when a speaker corrects an earlier assertion."
            )
        if failure_counts.get(FailureMode.DISENTANGLEMENT.value, 0) > 0:
            recommendations.append(
                "Disentanglement Gap: The model recognizes semantic meaning but cannot infer causal origin. "
                "Compare the query against the support set baseline: if the lexical mapping differs from the root language, "
                "check whether that divergence occurs in other historical periods of that community vs across all communities."
            )
        if failure_counts.get(FailureMode.COUNTERFACTUAL_BIAS.value, 0) > 0:
            recommendations.append(
                "Counterfactual Shortcut Bias Detected: The model is outputting identical predictions when context changes. "
                "Avoid relying on global prior probabilities; predictions must strictly condition on the active dialogue context."
            )
        if not recommendations:
            recommendations.append("Excellent performance across all diagnostic categories with minimal systemic weaknesses.")

        return DiagnosticReport(
            total_episodes=total,
            total_errors=total_errors,
            exact_match_accuracy=round(exact_count / total * 100.0, 2),
            failure_counts=dict(failure_counts),
            failure_percentages=failure_percentages,
            slot_error_counts=dict(slot_error_counts),
            phenomenon_error_rates=phenomenon_error_rates,
            level_error_rates=level_error_rates,
            episode_diagnostics=diagnostics,
            recommendations=recommendations
        )
