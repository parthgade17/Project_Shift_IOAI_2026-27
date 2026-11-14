"""
Project SHIFT: Counterfactual Context Sensitivity Experiment.
IOAI 2027 Scientific Benchmark Experiment Suite (EXP-06).

Evaluates models on paired counterfactual dialogue episodes (datasets/test_counterfactual.jsonl).
Each pair (Branch A and Branch B) shares the exact same query utterance U_q and grammatical rules,
differing exclusively in conversational dialogue antecedents.
Measures:
  - Branch A Accuracy
  - Branch B Accuracy
  - Joint Pair Accuracy (both branches solved simultaneously)
  - Perturbation Sensitivity (model output changes when context ground truth changes)
  - Composite Consistency Score
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shift.serialize import load_episodes_jsonl
from shift.evaluator import Evaluator
from baselines.majority import MajorityBaseline
from baselines.surface_lookup import SurfaceLookupBaseline
from baselines.statistical import StatisticalLearner
from baselines.rule_based import RuleBasedLearner


def run_counterfactual_test_experiment(
    cf_path: str = "datasets/test_counterfactual.jsonl",
    output_dir: Path = None,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" [EXP-06] COUNTERFACTUAL CONTEXT SENSITIVITY EXPERIMENT")
    print(f" Paired counterfactual split: {cf_path}")
    print("=" * 72)

    cf_episodes = load_episodes_jsonl(cf_path)
    cf_gt = {ep.episode_id: ep.query_label for ep in cf_episodes}
    cf_dict = {ep.episode_id: ep for ep in cf_episodes}

    baselines = [
        ("Majority Baseline", MajorityBaseline()),
        ("Surface Lookup Baseline", SurfaceLookupBaseline()),
        ("Statistical Naive Bayes", StatisticalLearner()),
        ("Inductive Rule Learner", RuleBasedLearner()),
    ]

    results: Dict[str, Any] = {}

    for name, model in baselines:
        print(f"\n--> Model: {name}")
        preds = {ep.episode_id: model.predict_episode(ep) for ep in cf_episodes}
        m = Evaluator.evaluate(preds, cf_gt, cf_dict)

        cf_metrics = m.get("counterfactual_consistency", {})
        exact_match = m["exact_match_accuracy"]
        joint_acc = cf_metrics.get("joint_accuracy", 0.0)
        sensitivity = cf_metrics.get("perturbation_sensitivity", 0.0)
        consistency = cf_metrics.get("consistency_score", 0.0)
        branch_a = cf_metrics.get("branch_a_accuracy", 0.0)
        branch_b = cf_metrics.get("branch_b_accuracy", 0.0)
        total_pairs = cf_metrics.get("total_pairs", 0)

        results[name] = {
            "exact_match_accuracy": exact_match,
            "total_pairs": total_pairs,
            "branch_a_accuracy": branch_a,
            "branch_b_accuracy": branch_b,
            "joint_accuracy": joint_acc,
            "perturbation_sensitivity": sensitivity,
            "composite_consistency_score": consistency,
        }

        print(f"    * Exact Match Accuracy     : {exact_match:5.1f}%")
        print(f"    * Branch A Accuracy        : {branch_a:5.1f}%")
        print(f"    * Branch B Accuracy        : {branch_b:5.1f}%")
        print(f"    * Joint Pair Accuracy      : {joint_acc:5.1f}%")
        print(f"    * Perturbation Sensitivity : {sensitivity:5.1f}%")
        print(f"    * Consistency Score        : {consistency:5.1f}% (N={total_pairs} pairs)")

    summary_output = {
        "experiment_id": "EXP-06",
        "description": "Counterfactual Context Sensitivity and Joint Pair Consistency Evaluation",
        "num_episodes": len(cf_episodes),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }

    out_file = output_dir / "counterfactual_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    print(f"\n[OK] Results saved to {out_file}")

    return summary_output


if __name__ == "__main__":
    run_counterfactual_test_experiment()
