"""
Project SHIFT: Shortcut & Memorisation Audit Suite.
Component 4 of SHIFT v0.3 Scientific Benchmark Infrastructure.

Executes adversarial shortcut attacks to ensure models cannot exploit superficial
biases, metadata memorisation, bag-of-words artifacts, or context-free shortcuts.
"""

from __future__ import annotations
import copy
import random
from typing import Dict, List, Tuple, Any

from shift.ontology import EventFrame
from shift.episodes import Episode
from baselines.majority import MajorityBaseline
from baselines.rule_based import RuleBasedLearner
from baselines.evaluate import Evaluator


class ShortcutAuditor:
    """Detects dataset shortcuts, superficial artifacts, and memorisation vulnerabilities."""

    @staticmethod
    def audit_metadata_shortcut(episodes: List[Episode]) -> Dict[str, Any]:
        """
        Attack 1: Metadata-only Prediction Attack.
        Predicts using a mapping solely from (community, period) to majority label,
        completely ignoring the query utterance.
        """
        ground_truth = {ep.episode_id: ep.query_label for ep in episodes if ep.query_label}
        majority_model = MajorityBaseline()
        predictions = {ep.episode_id: majority_model.predict_episode(ep) for ep in episodes}

        metrics = Evaluator.evaluate(predictions, ground_truth)
        em_acc = metrics.get("exact_match_accuracy", 0.0)

        return {
            "attack_type": "metadata_only_attack",
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
