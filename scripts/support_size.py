"""
Project SHIFT: Few-Shot Support Size Adaptation Experiment.
Empirically tests model adaptation efficiency across support sizes k in {0, 4, 8, 16, 32}
over multiple random seeds to measure adaptation slope and sample efficiency.
"""

from __future__ import annotations
import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shift.episodes import EpisodeGenerator
from shift.evaluator import Evaluator
from baselines.majority import MajorityBaseline
from baselines.surface_lookup import SurfaceLookupBaseline
from baselines.statistical import StatisticalLearner
from baselines.rule_based import RuleBasedLearner


def run_support_size_experiment(
    k_values: List[int] = [0, 4, 8, 16, 32],
    seeds: List[int] = [42, 101, 2027],
    batch_size: int = 30,
    output_path: str = "results/support_size_results.json"
) -> Dict[str, Any]:
    print("=" * 70)
    print(" PROJECT SHIFT: Few-Shot Support Adaptation Experiment")
    print("=" * 70)

    models = [
        ("Inductive Rule Learner", RuleBasedLearner()),
        ("Statistical Naive Bayes", StatisticalLearner()),
        ("Surface Lookup Baseline", SurfaceLookupBaseline()),
        ("Majority Baseline", MajorityBaseline()),
    ]

    # Structure: results[model_name][k] = [seed1_acc, seed2_acc, ...]
    raw_scores: Dict[str, Dict[int, List[float]]] = {
        m[0]: {k: [] for k in k_values} for m in models
    }

    for seed in seeds:
        print(f"\n--- Running Seed {seed} ---")
        generator = EpisodeGenerator(base_seed=seed)

        for k in k_values:
            episodes = generator.generate_batch(count=batch_size, k_support=k, id_prefix=f"s{seed}_k{k}")
            gt = {ep.episode_id: ep.query_label for ep in episodes}
            ep_dict = {ep.episode_id: ep for ep in episodes}

            for name, model in models:
                preds = {ep.episode_id: model.predict_episode(ep) for ep in episodes}
                eval_res = Evaluator.evaluate(preds, gt, ep_dict)
                acc = eval_res["exact_match_accuracy"]
                raw_scores[name][k].append(acc)

    # Compute mean and standard deviation
    summary_results: Dict[str, Any] = {
        "k_values": k_values,
        "seeds": seeds,
        "models": {}
    }

    print("\n" + "=" * 75)
    print(f"{'Model Architecture':<28} | {'k=0 (Zero)':<11} | {'k=4':<11} | {'k=8':<11} | {'k=16':<11} | {'k=32':<11}")
    print("-" * 75)

    for name, _ in models:
        summary_results["models"][name] = {}
        row_str = f"{name:<28} |"

        for k in k_values:
            scores = raw_scores[name][k]
            mean_acc = sum(scores) / len(scores)
            variance = sum((s - mean_acc) ** 2 for s in scores) / len(scores) if len(scores) > 1 else 0.0
            std_dev = math.sqrt(variance)

            summary_results["models"][name][f"k={k}"] = {
                "mean": round(mean_acc, 2),
                "std": round(std_dev, 2),
                "runs": scores
            }
            cell = f"{mean_acc:5.1f}+/-{std_dev:3.1f}%"
            row_str += f" {cell:<11} |"

        print(row_str)

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)

    print(f"\n[OK] Results exported to {output_path}\n")
    return summary_results


if __name__ == "__main__":
    run_support_size_experiment()
