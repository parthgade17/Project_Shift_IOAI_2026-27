"""
Project SHIFT: Evaluation Engine and Metric Computation.
Milestone 11 Deliverable for IOAI 2027 Task Development.

Computes exact event-frame accuracy, slot-level micro/macro accuracies,
semantic drift disentanglement accuracy, counterfactual consistency,
progressive difficulty levels (Levels 1-5), and runtime efficiency.
"""

from __future__ import annotations
import time
import argparse
import json
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict

from shift.ontology import EventFrame
from shift.episodes import Episode
from shift.serialize import load_episodes_jsonl, load_ground_truth_labels


class Evaluator:
    """Computes comprehensive evaluation metrics for Project SHIFT."""

    @staticmethod
    def evaluate(
        predictions: Dict[str, EventFrame],
        ground_truth: Dict[str, EventFrame],
        episodes_by_id: Optional[Dict[str, Episode]] = None
    ) -> Dict[str, Any]:
        """
        Evaluates predictions against ground truth.
        Returns detailed metric breakdown.
        """
        total = len(ground_truth)
        if total == 0:
            return {"error": "Ground truth is empty"}

        exact_matches = 0
        agent_matches = 0
        action_matches = 0
        object_matches = 0
        tense_matches = 0
        polarity_matches = 0
        ref_matches = 0
        disentanglement_matches = 0

        phenomenon_stats = {}
        level_stats = defaultdict(lambda: {"total": 0, "exact": 0})
        cf_pairs = defaultdict(dict)

        for ep_id, true_frame in ground_truth.items():
            pred_frame = predictions.get(ep_id)
            if pred_frame is None:
                continue

            is_exact = true_frame.exact_match(pred_frame)
            if is_exact:
                exact_matches += 1

            if true_frame.agent == pred_frame.agent: agent_matches += 1
            if true_frame.action == pred_frame.action: action_matches += 1
            if true_frame.object == pred_frame.object: object_matches += 1
            if true_frame.tense == pred_frame.tense: tense_matches += 1
            if true_frame.polarity == pred_frame.polarity: polarity_matches += 1
            if true_frame.resolved_references == pred_frame.resolved_references: ref_matches += 1

            # Disentanglement attribution check
            if true_frame.drift_attribution and pred_frame.drift_attribution:
                if true_frame.drift_attribution == pred_frame.drift_attribution:
                    disentanglement_matches += 1

            # Context & Metadata tracking
            if episodes_by_id and ep_id in episodes_by_id:
                ep = episodes_by_id[ep_id]
                phenom = ep.query.phenomenon
                if phenom not in phenomenon_stats:
                    phenomenon_stats[phenom] = {"total": 0, "exact": 0}
                phenomenon_stats[phenom]["total"] += 1
                if is_exact:
                    phenomenon_stats[phenom]["exact"] += 1

                # Difficulty Level (Levels 1 to 5)
                lvl = ep.difficulty_level
                level_stats[lvl]["total"] += 1
                if is_exact:
                    level_stats[lvl]["exact"] += 1

                # Counterfactual Pair tracking
                if ep.counterfactual_pair_id:
                    cf_pairs[ep.counterfactual_pair_id][ep_id] = is_exact

        # Counterfactual consistency calculation
        cf_consistent_pairs = 0
        total_cf_pairs = len(cf_pairs)
        for pair_id, branches in cf_pairs.items():
            if len(branches) == 2 and all(branches.values()):
                cf_consistent_pairs += 1

        cf_consistency_score = round((cf_consistent_pairs / total_cf_pairs * 100.0), 2) if total_cf_pairs else 100.0

        metrics = {
            "total_episodes": total,
            "predictions_count": len(predictions),
            "exact_match_accuracy": round((exact_matches / total) * 100.0, 2),
            "slot_accuracies": {
                "agent": round((agent_matches / total) * 100.0, 2),
                "action": round((action_matches / total) * 100.0, 2),
                "object": round((object_matches / total) * 100.0, 2),
                "tense": round((tense_matches / total) * 100.0, 2),
                "polarity": round((polarity_matches / total) * 100.0, 2),
                "resolved_references": round((ref_matches / total) * 100.0, 2),
            },
            "disentanglement_attribution_accuracy": round((disentanglement_matches / total) * 100.0, 2),
            "counterfactual_consistency": {
                "total_pairs": total_cf_pairs,
                "consistent_pairs": cf_consistent_pairs,
                "consistency_score": cf_consistency_score
            },
            "progressive_levels": {
                f"Level {lvl}": {
                    "total": level_stats[lvl]["total"],
                    "accuracy": round((level_stats[lvl]["exact"] / level_stats[lvl]["total"]) * 100.0, 2)
                    if level_stats[lvl]["total"] else 0.0
                }
                for lvl in sorted(level_stats.keys())
            },
            "phenomenon_breakdown": {
                k: {
                    "total": v["total"],
                    "accuracy": round((v["exact"] / v["total"]) * 100.0, 2) if v["total"] else 0.0
                }
                for k, v in phenomenon_stats.items()
            }
        }
        return metrics

    @staticmethod
    def print_report(metrics: Dict[str, Any], title: str = "Evaluation Report") -> None:
        """Prints a human-readable console report."""
        print(f"\n=================== {title} ===================")
        print(f"Total Episodes Evaluated : {metrics.get('total_episodes', 0)}")
        print(f"Exact Frame Match Acc    : {metrics.get('exact_match_accuracy', 0.0)}%")
        print(f"Disentanglement Acc      : {metrics.get('disentanglement_attribution_accuracy', 0.0)}%")
        print("-----------------------------------------------------")
        print("Slot Accuracies:")
        for slot, acc in metrics.get("slot_accuracies", {}).items():
            print(f"  • {slot:<22}: {acc}%")

        if metrics.get("counterfactual_consistency", {}).get("total_pairs", 0) > 0:
            cf = metrics["counterfactual_consistency"]
            print("-----------------------------------------------------")
            print(f"Counterfactual Consistency: {cf['consistency_score']}% ({cf['consistent_pairs']}/{cf['total_pairs']} pairs)")

        if metrics.get("progressive_levels"):
            print("-----------------------------------------------------")
            print("Progressive Difficulty Levels:")
            for lvl, stats in metrics["progressive_levels"].items():
                print(f"  • {lvl:<22}: {stats['accuracy']}% (N={stats['total']})")

        if metrics.get("phenomenon_breakdown"):
            print("-----------------------------------------------------")
            print("Phenomenon Breakdown:")
            for phenom, stats in metrics["phenomenon_breakdown"].items():
                print(f"  • {phenom:<22}: {stats['accuracy']}% (N={stats['total']})")
        print("=====================================================\n")


def run_evaluation_cli():
    """Command-line runner for benchmarking systems."""
    parser = argparse.ArgumentParser(description="Evaluate Project SHIFT predictions.")
    parser.add_argument("--dataset", required=True, help="Path to episodes JSONL file")
    parser.add_argument("--labels", help="Path to ground truth labels JSON (optional if dataset has query_label)")
    parser.add_argument("--baseline", choices=["majority", "lookup", "statistical", "rule"], help="Run a built-in baseline")
    parser.add_argument("--predictions", help="Path to predictions JSON file (if not running built-in baseline)")
    parser.add_argument("--output", help="Optional path to save metrics JSON")

    args = parser.parse_args()

    episodes = load_episodes_jsonl(args.dataset)
    ep_dict = {ep.episode_id: ep for ep in episodes}

    if args.labels:
        ground_truth = load_ground_truth_labels(args.labels)
    else:
        ground_truth = {ep.episode_id: ep.query_label for ep in episodes if ep.query_label is not None}

    predictions: Dict[str, EventFrame] = {}

    if args.baseline == "majority":
        from baselines.majority import MajorityBaseline
        m = MajorityBaseline()
        start = time.time()
        for ep in episodes:
            predictions[ep.episode_id] = m.predict_episode(ep)
        elapsed = time.time() - start
        print(f"Inference complete: {len(episodes)} episodes in {elapsed*1000:.2f} ms ({elapsed/len(episodes)*1000:.2f} ms/ep)")

    elif args.baseline == "lookup":
        from baselines.surface_lookup import SurfaceLookupBaseline
        m = SurfaceLookupBaseline()
        start = time.time()
        for ep in episodes:
            predictions[ep.episode_id] = m.predict_episode(ep)
        elapsed = time.time() - start
        print(f"Inference complete: {len(episodes)} episodes in {elapsed*1000:.2f} ms ({elapsed/len(episodes)*1000:.2f} ms/ep)")

    elif args.baseline == "statistical":
        from baselines.statistical import StatisticalLearner
        m = StatisticalLearner()
        start = time.time()
        for ep in episodes:
            predictions[ep.episode_id] = m.predict_episode(ep)
        elapsed = time.time() - start
        print(f"Inference complete: {len(episodes)} episodes in {elapsed*1000:.2f} ms ({elapsed/len(episodes)*1000:.2f} ms/ep)")

    elif args.baseline == "rule":
        from baselines.rule_based import RuleBasedLearner
        m = RuleBasedLearner()
        start = time.time()
        for ep in episodes:
            predictions[ep.episode_id] = m.predict_episode(ep)
        elapsed = time.time() - start
        print(f"Inference complete: {len(episodes)} episodes in {elapsed*1000:.2f} ms ({elapsed/len(episodes)*1000:.2f} ms/ep)")

    elif args.predictions:
        with open(args.predictions, "r", encoding="utf-8") as f:
            raw = json.load(f)
        predictions = {k: EventFrame.from_dict(v) for k, v in raw.items()}
    else:
        raise ValueError("Must specify either --baseline [majority|lookup|statistical|rule] or --predictions path.json")

    metrics = Evaluator.evaluate(predictions, ground_truth, ep_dict)
    Evaluator.print_report(metrics, title=f"Results: {args.baseline or 'Custom Model'}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    run_evaluation_cli()
