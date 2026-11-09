"""
Project SHIFT: Robustness and Shortcut Testing Suite.
Milestone 12 Deliverable for IOAI 2027 Task Development.

Audits whether trivial shortcuts (metadata memorization, bag-of-words,
context ablation, or support removal) can solve the task without genuine adaptation.
"""

from __future__ import annotations
import copy
from typing import Dict, List, Tuple, Any

from shift.ontology import EventFrame
from shift.episodes import Episode
from baselines.majority import MajorityBaseline
from baselines.rule_based import RuleBasedLearner
from baselines.evaluate import Evaluator


class ShortcutAuditor:
    """Executes counterfactual stress tests and shortcut ablation attacks."""

    @staticmethod
    def audit_metadata_shortcut(episodes: List[Episode]) -> Dict[str, Any]:
        """
        Metadata-only prediction attack:
        Predicts using a mapping solely from (community, period) to most common label.
        Measures if the dataset is vulnerable to metadata memorization.
        """
        ground_truth = {ep.episode_id: ep.query_label for ep in episodes if ep.query_label}
        predictions: Dict[str, EventFrame] = {}

        # Run majority baseline (which ignores the query utterance entirely)
        majority_model = MajorityBaseline()
        for ep in episodes:
            predictions[ep.episode_id] = majority_model.predict_episode(ep)

        metrics = Evaluator.evaluate(predictions, ground_truth)
        return {
            "attack_type": "metadata_only_attack",
            "exact_match_accuracy": metrics["exact_match_accuracy"],
            "vulnerability_detected": metrics["exact_match_accuracy"] > 25.0
        }

    @staticmethod
    def audit_support_ablation(episodes: List[Episode]) -> Dict[str, Any]:
        """
        Support removal attack (k = 0):
        Strips all support examples and evaluates if the rule-based learner degrades.
        A well-designed challenge MUST show a severe degradation when support is removed.
        """
        ground_truth = {ep.episode_id: ep.query_label for ep in episodes if ep.query_label}
        ablated_episodes = []
        for ep in episodes:
            ep_copy = copy.deepcopy(ep)
            ep_copy.support_set = [] # Strip support
            ablated_episodes.append(ep_copy)

        learner = RuleBasedLearner()
        predictions = {ep.episode_id: learner.predict_episode(ep) for ep in ablated_episodes}
        metrics = Evaluator.evaluate(predictions, ground_truth)

        return {
            "attack_type": "support_ablation_attack (k=0)",
            "exact_match_accuracy": metrics["exact_match_accuracy"],
            "support_dependency_confirmed": metrics["exact_match_accuracy"] <= 30.0
        }

    @staticmethod
    def audit_context_ablation(episodes: List[Episode]) -> Dict[str, Any]:
        """
        Context ablation attack:
        Strips dialogue context turns for coreference queries.
        Coreference accuracy should drop significantly without discourse antecedents.
        """
        coref_episodes = [ep for ep in episodes if ep.query.phenomenon in ("coreference", "ellipsis")]
        if not coref_episodes:
            return {"attack_type": "context_ablation_attack", "note": "No coreference episodes in batch"}

        ground_truth = {ep.episode_id: ep.query_label for ep in coref_episodes if ep.query_label}
        ablated = []
        for ep in coref_episodes:
            ep_c = copy.deepcopy(ep)
            ep_c.query.context = [] # Strip context
            ablated.append(ep_c)

        learner = RuleBasedLearner()
        predictions = {ep.episode_id: learner.predict_episode(ep) for ep in ablated}
        metrics = Evaluator.evaluate(predictions, ground_truth)

        return {
            "attack_type": "context_ablation_attack",
            "exact_match_accuracy": metrics["exact_match_accuracy"],
            "context_dependency_confirmed": metrics["exact_match_accuracy"] < 35.0
        }

    @staticmethod
    def audit_counterfactual_sensitivity(episodes: List[Episode]) -> Dict[str, Any]:
        """
        Counterfactual pair audit:
        Verifies that when two episodes share identical query utterance but different contexts,
        the model correctly adapts its interpretation rather than predicting static outputs.
        """
        cf_episodes = [ep for ep in episodes if ep.counterfactual_pair_id]
        if not cf_episodes:
            return {"attack_type": "counterfactual_sensitivity", "note": "No counterfactual pairs in batch"}

        ground_truth = {ep.episode_id: ep.query_label for ep in cf_episodes if ep.query_label}
        ep_dict = {ep.episode_id: ep for ep in cf_episodes}
        learner = RuleBasedLearner()
        predictions = {ep.episode_id: learner.predict_episode(ep) for ep in cf_episodes}

        metrics = Evaluator.evaluate(predictions, ground_truth, ep_dict)
        cf_score = metrics.get("counterfactual_consistency", {}).get("consistency_score", 0.0)

        return {
            "attack_type": "counterfactual_sensitivity",
            "consistency_score": cf_score,
            "context_sensitivity_confirmed": cf_score >= 80.0
        }

    @classmethod
    def run_full_robustness_suite(cls, episodes: List[Episode]) -> Dict[str, Any]:
        """Runs all shortcut audits and compiles a summary report."""
        meta_res = cls.audit_metadata_shortcut(episodes)
        supp_res = cls.audit_support_ablation(episodes)
        ctx_res = cls.audit_context_ablation(episodes)
        cf_res = cls.audit_counterfactual_sensitivity(episodes)

        all_passed = (
            not meta_res["vulnerability_detected"] and
            supp_res["support_dependency_confirmed"] and
            ctx_res.get("context_dependency_confirmed", True)
        )

        return {
            "overall_robustness_passed": all_passed,
            "metadata_attack": meta_res,
            "support_ablation_attack": supp_res,
            "context_ablation_attack": ctx_res,
            "counterfactual_sensitivity": cf_res
        }
