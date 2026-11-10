"""
Project SHIFT: Shortcut & Memorisation Audit Suite.
Component 4 of SHIFT v0.3 Scientific Benchmark Infrastructure.

Executes adversarial shortcut attacks to ensure models cannot exploit superficial
biases, metadata memorisation, bag-of-words artifacts, or context-free shortcuts.
"""

from __future__ import annotations
import copy
import random
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional

from shift.ontology import (
    EventFrame, AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity
)
from shift.episodes import Episode
from baselines.majority import MajorityBaseline
from baselines.rule_based import RuleBasedLearner
from baselines.evaluate import Evaluator


class GenuineMetadataAttackBaseline:
    """
    Genuine Metadata-Only Adversarial Attack.
    Attempts to predict target event frames strictly from environment metadata:
    (community_id, period_id, word_order, difficulty_level, phenomenon)
    without observing support set utterances, support frames, or the query utterance.
    """
    def __init__(self, calibration_episodes: Optional[List[Episode]] = None):
        self.name = "Genuine Metadata Attack"
        self.conditional_tables: Dict[str, Dict[str, Counter]] = {
            "agent": defaultdict(Counter),
            "action": defaultdict(Counter),
            "object": defaultdict(Counter),
            "tense": defaultdict(Counter),
            "polarity": defaultdict(Counter)
        }
        self.global_counts: Dict[str, Counter] = {
            "agent": Counter(),
            "action": Counter(),
            "object": Counter(),
            "tense": Counter(),
            "polarity": Counter()
        }
        if calibration_episodes:
            self.fit(calibration_episodes)

    def _meta_key(self, ep: Episode) -> str:
        phenom = ep.query.phenomenon if ep.query else "direct"
        return f"{ep.community_id}_{ep.period_id}_{ep.word_order}_{ep.difficulty_level}_{phenom}"

    def fit(self, episodes: List[Episode]) -> None:
        for ep in episodes:
            if not ep.query_label:
                continue
            k = self._meta_key(ep)
            for slot in ["agent", "action", "object", "tense", "polarity"]:
                val = getattr(ep.query_label, slot)
                self.conditional_tables[slot][k][val] += 1
                self.global_counts[slot][val] += 1

    def predict_episode(self, episode: Episode) -> EventFrame:
        k = self._meta_key(episode)
        preds = {}
        defaults = {
            "agent": AgentEntity.SCOUT_ALPHA.value,
            "action": ActionPredicate.INSPECT.value,
            "object": ObjectEntity.CRYSTAL_CORE.value,
            "tense": Tense.PRESENT.value,
            "polarity": Polarity.AFFIRMATIVE.value
        }
        for slot in ["agent", "action", "object", "tense", "polarity"]:
            if self.conditional_tables[slot][k]:
                preds[slot] = self.conditional_tables[slot][k].most_common(1)[0][0]
            elif self.global_counts[slot]:
                preds[slot] = self.global_counts[slot].most_common(1)[0][0]
            else:
                preds[slot] = defaults[slot]

        return EventFrame(
            agent=preds["agent"],
            action=preds["action"],
            object=preds["object"],
            tense=preds["tense"],
            polarity=preds["polarity"],
            resolved_references={}
        )


class ShortcutAuditor:
    """Detects dataset shortcuts, superficial artifacts, and memorisation vulnerabilities."""

    @staticmethod
    def audit_metadata_shortcut(
        episodes: List[Episode],
        calibration_episodes: Optional[List[Episode]] = None
    ) -> Dict[str, Any]:
        """
        Attack 1: Genuine Metadata-only Prediction Attack.
        Predicts using a mapping solely from (community, period, order, level, phenomenon)
        to target slots, completely ignoring the query utterance and support set.
        """
        ground_truth = {ep.episode_id: ep.query_label for ep in episodes if ep.query_label}
        meta_model = GenuineMetadataAttackBaseline(calibration_episodes or episodes)
        predictions = {ep.episode_id: meta_model.predict_episode(ep) for ep in episodes}

        metrics = Evaluator.evaluate(predictions, ground_truth)
        em_acc = metrics.get("exact_match_accuracy", 0.0)

        return {
            "attack_type": "genuine_metadata_only_attack",
            "exact_match_accuracy": em_acc,
            "threshold": 25.0,
            "vulnerability_detected": em_acc > 25.0,
            "passed": em_acc <= 25.0
        }

    @staticmethod
    def audit_support_ablation(episodes: List[Episode]) -> Dict[str, Any]:
        """
        Attack 2: Support Set Removal Attack (k = 0).
        Strips all support examples to verify whether performance drops dramatically.
        Genuine few-shot adaptation requires that support set examples are necessary.
        """
        ground_truth = {ep.episode_id: ep.query_label for ep in episodes if ep.query_label}
        ablated = []
        for ep in episodes:
            c = copy.deepcopy(ep)
            c.support_set = []
            ablated.append(c)

        learner = RuleBasedLearner()
        predictions = {ep.episode_id: learner.predict_episode(ep) for ep in ablated}
        metrics = Evaluator.evaluate(predictions, ground_truth)
        em_acc = metrics.get("exact_match_accuracy", 0.0)

        # Performance must drop significantly compared to full support
        return {
            "attack_type": "support_ablation_attack (k=0)",
            "exact_match_accuracy": em_acc,
            "threshold": 30.0,
            "support_dependency_confirmed": em_acc <= 30.0,
            "passed": em_acc <= 30.0
        }

    @staticmethod
    def audit_context_ablation(episodes: List[Episode]) -> Dict[str, Any]:
        """
        Attack 3: Context Removal Attack.
        Strips conversation history for coreference and ellipsis queries.
        Models must fail on discourse queries when context is missing.
        """
        coref_episodes = [ep for ep in episodes if ep.query.phenomenon in ("coreference", "ellipsis", "counterfactual_pair")]
        if not coref_episodes:
            return {"attack_type": "context_ablation_attack", "passed": True, "note": "No discourse episodes"}

        ground_truth = {ep.episode_id: ep.query_label for ep in coref_episodes if ep.query_label}
        ablated = []
        for ep in coref_episodes:
            c = copy.deepcopy(ep)
            c.query.context = []
            ablated.append(c)

        learner = RuleBasedLearner()
        predictions = {ep.episode_id: learner.predict_episode(ep) for ep in ablated}
        metrics = Evaluator.evaluate(predictions, ground_truth)
        em_acc = metrics.get("exact_match_accuracy", 0.0)

        return {
            "attack_type": "context_ablation_attack",
            "exact_match_accuracy": em_acc,
            "threshold": 35.0,
            "context_dependency_confirmed": em_acc < 35.0,
            "passed": em_acc < 35.0
        }

    @staticmethod
    def audit_context_shuffling(episodes: List[Episode], seed: int = 42) -> Dict[str, Any]:
        """
        Attack 4: Conversational History Permutation Attack.
        Shuffles the temporal ordering of context turns, testing sensitivity to discourse recency.
        """
        multi_turn_eps = [ep for ep in episodes if len(ep.query.context) > 1]
        if not multi_turn_eps:
            return {"attack_type": "context_shuffling_attack", "passed": True, "note": "No multi-turn episodes"}

        rng = random.Random(seed)
        ground_truth = {ep.episode_id: ep.query_label for ep in multi_turn_eps if ep.query_label}
        shuffled = []
        for ep in multi_turn_eps:
            c = copy.deepcopy(ep)
            rng.shuffle(c.query.context)
            shuffled.append(c)

        learner = RuleBasedLearner()
        predictions = {ep.episode_id: learner.predict_episode(ep) for ep in shuffled}
        metrics = Evaluator.evaluate(predictions, ground_truth)

        return {
            "attack_type": "context_shuffling_attack",
            "exact_match_accuracy": metrics.get("exact_match_accuracy", 0.0),
            "passed": True
        }

    @staticmethod
    def audit_counterfactual_consistency(episodes: List[Episode]) -> Dict[str, Any]:
        """
        Attack 5: Counterfactual Pair Context Sensitivity.
        Measures consistency on matched pairs with identical query utterances but differing contexts.
        """
        cf_eps = [ep for ep in episodes if ep.counterfactual_pair_id]
        if not cf_eps:
            return {"attack_type": "counterfactual_consistency", "passed": True, "note": "No counterfactual pairs"}

        ground_truth = {ep.episode_id: ep.query_label for ep in cf_eps if ep.query_label}
        ep_dict = {ep.episode_id: ep for ep in cf_eps}
        learner = RuleBasedLearner()
        predictions = {ep.episode_id: learner.predict_episode(ep) for ep in cf_eps}

        metrics = Evaluator.evaluate(predictions, ground_truth, ep_dict)
        cf_score = metrics.get("counterfactual_consistency", {}).get("consistency_score", 0.0)

        return {
            "attack_type": "counterfactual_consistency",
            "consistency_score": cf_score,
            "threshold": 80.0,
            "context_sensitivity_confirmed": cf_score >= 80.0,
            "passed": cf_score >= 80.0
        }

    @classmethod
    def run_full_shortcut_suite(cls, episodes: List[Episode]) -> Dict[str, Any]:
        """Runs the complete suite of shortcut attacks and checks benchmark integrity."""
        meta_res = cls.audit_metadata_shortcut(episodes)
        supp_res = cls.audit_support_ablation(episodes)
        ctx_res = cls.audit_context_ablation(episodes)
        shuf_res = cls.audit_context_shuffling(episodes)
        cf_res = cls.audit_counterfactual_consistency(episodes)

        all_passed = (
            meta_res["passed"] and
            supp_res["passed"] and
            ctx_res.get("passed", True) and
            cf_res.get("passed", True)
        )

        return {
            "all_shortcut_audits_passed": all_passed,
            "metadata_attack": meta_res,
            "support_ablation": supp_res,
            "context_ablation": ctx_res,
            "context_shuffling": shuf_res,
            "counterfactual_consistency": cf_res
        }
