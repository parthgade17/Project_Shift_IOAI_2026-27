"""
Project SHIFT: Baseline Comparison Experiment Runner.
Compares four baseline model families:
1. Majority Class Baseline
2. Surface Lookup (Nearest-Neighbour Token Overlap)
3. Statistical Naive Bayes (Maximum Likelihood Estimation)
4. Inductive Rule Learner (Constituent Induction & Disentanglement)

Reports Exact Match, Slot F1s, Disentanglement Accuracy, Reference Resolution,
Counterfactual Consistency, and Latency per episode.
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shift.serialize import load_episodes_jsonl
from shift.evaluator import Evaluator
from baselines.majority import MajorityBaseline
from baselines.surface_lookup import SurfaceLookupBaseline
from baselines.statistical import StatisticalLearner
from baselines.rule_based import RuleBasedLearner


def run_baseline_comparison(
    test_path: str = "datasets/hidden_test_full_reference.jsonl",
    output_path: str = "results/baseline_comparison_results.json"
) -> Dict[str, Any]:
    print("=" * 70)
    print(" PROJECT SHIFT: Baseline Systems Comparative Experiment")
    print("=" * 70)

    episodes = load_episodes_jsonl(test_path)
    gt = {ep.episode_id: ep.query_label for ep in episodes}
    ep_dict = {ep.episode_id: ep for ep in episodes}

    models = [
        ("Majority Baseline", MajorityBaseline()),
        ("Surface Lookup Baseline", SurfaceLookupBaseline()),
        ("Statistical Naive Bayes", StatisticalLearner()),
        ("Inductive Rule Learner", RuleBasedLearner()),
    ]

    results: Dict[str, Any] = {}

    print(f"\nEvaluating on {len(episodes)} Hidden Test Episodes:")
    print(f"{'Model Architecture':<30} | {'Exact Match':<12} | {'Composite':<10} | {'Disentangle':<12} | {'Latency':<10}")
    print("-" * 84)

    for name, model in models:
        # Fit models that support training
        if hasattr(model, "fit"):
            train_eps = load_episodes_jsonl("datasets/train.jsonl")
            model.fit(train_eps)

        start_t = time.perf_counter()
        preds = {ep.episode_id: model.predict_episode(ep) for ep in episodes}
        total_time = time.perf_counter() - start_t
        avg_latency_ms = (total_time / len(episodes)) * 1000.0

        eval_res = Evaluator.evaluate(
            predictions=preds,
            ground_truth=gt,
            episodes_by_id=ep_dict
        )
        eval_res["latency_ms_per_episode"] = round(avg_latency_ms, 3)
        results[name] = eval_res

        em = eval_res["exact_match_accuracy"]
        comp = eval_res["composite_competition_score"]
        dis = eval_res["disentanglement_accuracy"]
        print(f"{name:<30} | {em:10.2f}% | {comp:10.2f} | {dis:10.2f}% | {avg_latency_ms:7.3f} ms")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Results exported to {output_path}\n")
    return results


if __name__ == "__main__":
    run_baseline_comparison()
