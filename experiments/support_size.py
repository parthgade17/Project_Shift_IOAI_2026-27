"""
Project SHIFT: Few-Shot Adaptation Curve Experiment.
IOAI 2027 Scientific Benchmark Experiment Suite (EXP-01).

Systematically evaluates model performance across support sizes k in {0, 4, 8, 16, 32}
over multiple random seeds (42, 101, 2024). Measures exact match accuracy and slot F1,
computing mean and standard deviation to prove few-shot learning efficiency.
"""

from __future__ import annotations
import sys
import json
import time
import statistics
from pathlib import Path
from typing import Dict, List, Any

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shift.episodes import EpisodeGenerator
from shift.evaluator import Evaluator
from baselines.majority import MajorityBaseline
from baselines.surface_lookup import SurfaceLookupBaseline
from baselines.statistical import StatisticalLearner
from baselines.rule_based import RuleBasedLearner


def run_support_size_experiment(
    k_values: List[int] = None,
    seeds: List[int] = None,
    count_per_seed: int = 20,
    output_dir: Path = None,
) -> Dict[str, Any]:
    if k_values is None:
        k_values = [0, 4, 8, 16, 32]
    if seeds is None:
        seeds = [42, 101, 2024]
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" [EXP-01] FEW-SHOT ADAPTATION CURVE EXPERIMENT")
    print(f" Support sizes k: {k_values}")
    print(f" Random seeds   : {seeds} (N={count_per_seed} episodes/seed/k)")
    print("=" * 72)

    baseline_factories = [
        ("Majority Baseline", lambda: MajorityBaseline()),
        ("Surface Lookup Baseline", lambda: SurfaceLookupBaseline()),
        ("Statistical Naive Bayes", lambda: StatisticalLearner()),
        ("Inductive Rule Learner", lambda: RuleBasedLearner()),
    ]

    curve_results: Dict[str, Any] = {}

    for k in k_values:
        print(f"\n--> Evaluating Support Size k = {k:2d} across {len(seeds)} seeds...")
        seed_runs: Dict[str, List[float]] = {name: [] for name, _ in baseline_factories}
        slot_f1_runs: Dict[str, List[float]] = {name: [] for name, _ in baseline_factories}

        for seed in seeds:
            gen = EpisodeGenerator(base_seed=seed)
            batch = gen.generate_batch(
                count=count_per_seed,
                k_support=k,
                id_prefix=f"exp01_s{seed}_k{k}"
            )
            gt = {ep.episode_id: ep.query_label for ep in batch}
            ep_dict = {ep.episode_id: ep for ep in batch}

            for name, factory in baseline_factories:
                model = factory()
                preds = {ep.episode_id: model.predict_episode(ep) for ep in batch}
                metrics = Evaluator.evaluate(preds, gt, ep_dict)
                acc = metrics["exact_match_accuracy"]
                f1 = metrics["macro_slot_f1"]
                seed_runs[name].append(acc)
                slot_f1_runs[name].append(f1)

        k_entry: Dict[str, Any] = {}
        for name, _ in baseline_factories:
            accs = seed_runs[name]
            f1s = slot_f1_runs[name]
            mean_acc = round(statistics.mean(accs), 2)
            std_acc = round(statistics.stdev(accs), 2) if len(accs) > 1 else 0.0
            mean_f1 = round(statistics.mean(f1s), 2)
            std_f1 = round(statistics.stdev(f1s), 2) if len(f1s) > 1 else 0.0

            k_entry[name] = {
                "exact_match_mean": mean_acc,
                "exact_match_std": std_acc,
                "slot_f1_mean": mean_f1,
                "slot_f1_std": std_f1,
                "seed_runs": accs
            }
            print(f"    * {name:<26}: Acc = {mean_acc:5.1f}% (+/- {std_acc:4.1f}%), F1 = {mean_f1:5.1f}%")

        curve_results[f"k_{k}"] = k_entry

    summary_output = {
        "experiment_id": "EXP-01",
        "description": "Few-Shot Adaptation Curve across Support Sizes k in {0, 4, 8, 16, 32}",
        "k_values": k_values,
        "seeds": seeds,
        "count_per_seed": count_per_seed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": curve_results
    }

    out_file = output_dir / "support_size_curve.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    print(f"\n[OK] Results saved to {out_file}")

    return summary_output


if __name__ == "__main__":
    run_support_size_experiment()
