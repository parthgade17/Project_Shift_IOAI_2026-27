"""
Project SHIFT: Robustness & Adversarial Perturbation Experiment.
IOAI 2027 Scientific Benchmark Experiment Suite (EXP-07).

Evaluates model robustness across:
  1. Syntactic Word Order Variation: SOV vs SVO vs VSO.
  2. Pragmatic Negation / Inversion: Affirmative vs Negated queries.
  3. Morphological Case Variation: Synthetic archaic case affix (-a) vs Analytic forms.
  4. Lexical Distractor Noise: Robustness against non-critical dialogue particles.
"""

from __future__ import annotations
import sys
import copy
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


def run_robustness_test_experiment(
    test_path: str = "datasets/hidden_test_full_reference.jsonl",
    output_dir: Path = None,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" [EXP-07] ROBUSTNESS & PERTURBATION EXPERIMENT")
    print(f" Source dataset: {test_path}")
    print("=" * 72)

    test_episodes = load_episodes_jsonl(test_path)

    baselines = [
        ("Majority Baseline", MajorityBaseline()),
        ("Surface Lookup Baseline", SurfaceLookupBaseline()),
        ("Statistical Naive Bayes", StatisticalLearner()),
        ("Inductive Rule Learner", RuleBasedLearner()),
    ]

    # 1. Word Order Breakdown
    word_orders = ["SOV", "SVO", "VSO"]
    order_subsets = {
        wo: [ep for ep in test_episodes if ep.word_order == wo]
        for wo in word_orders
    }

    # 2. Polarity Breakdown
    affirmative_eps = [ep for ep in test_episodes if ep.query_label and ep.query_label.polarity == "affirmative"]
    negative_eps = [ep for ep in test_episodes if ep.query_label and ep.query_label.polarity == "negative"]

    # 3. Archaic Case Affix Breakdown
    archaic_eps = [ep for ep in test_episodes if "archaic" in ep.period_id.lower()]
    classical_contemp_eps = [ep for ep in test_episodes if "archaic" not in ep.period_id.lower()]

    results_by_model: Dict[str, Any] = {}

    for name, model in baselines:
        print(f"\n--> Model: {name}")
        model_data: Dict[str, Any] = {}

        # Evaluate Word Order robustness
        wo_metrics = {}
        for wo, eps in order_subsets.items():
            if eps:
                gt = {ep.episode_id: ep.query_label for ep in eps}
                ep_dict = {ep.episode_id: ep for ep in eps}
                preds = {ep.episode_id: model.predict_episode(ep) for ep in eps}
                m = Evaluator.evaluate(preds, gt, ep_dict)
                wo_metrics[wo] = {
                    "count": len(eps),
                    "exact_match_accuracy": m["exact_match_accuracy"],
                    "macro_slot_f1": m["macro_slot_f1"],
                }
                print(f"    * Word Order {wo:3s} (N={len(eps):2d}): Acc = {m['exact_match_accuracy']:5.1f}%")
        model_data["word_order_breakdown"] = wo_metrics

        # Evaluate Polarity / Pragmatic Negation
        pol_metrics = {}
        for p_name, eps in [("affirmative", affirmative_eps), ("negative", negative_eps)]:
            if eps:
                gt = {ep.episode_id: ep.query_label for ep in eps}
                ep_dict = {ep.episode_id: ep for ep in eps}
                preds = {ep.episode_id: model.predict_episode(ep) for ep in eps}
                m = Evaluator.evaluate(preds, gt, ep_dict)
                pol_metrics[p_name] = {
                    "count": len(eps),
                    "exact_match_accuracy": m["exact_match_accuracy"],
                    "polarity_accuracy": m.get("slot_accuracies", {}).get("polarity", 0.0),
                }
                print(f"    * Polarity {p_name:<11s} (N={len(eps):2d}): Acc = {m['exact_match_accuracy']:5.1f}%, Pol Acc = {m.get('slot_accuracies', {}).get('polarity', 0.0):5.1f}%")
        model_data["polarity_breakdown"] = pol_metrics

        # Evaluate Morphological Case
        case_metrics = {}
        for c_name, eps in [("archaic_case_inflected", archaic_eps), ("analytic_periods", classical_contemp_eps)]:
            if eps:
                gt = {ep.episode_id: ep.query_label for ep in eps}
                ep_dict = {ep.episode_id: ep for ep in eps}
                preds = {ep.episode_id: model.predict_episode(ep) for ep in eps}
                m = Evaluator.evaluate(preds, gt, ep_dict)
                case_metrics[c_name] = {
                    "count": len(eps),
                    "exact_match_accuracy": m["exact_match_accuracy"],
                    "patient_object_accuracy": m.get("slot_accuracies", {}).get("object", 0.0),
                }
                print(f"    * Morph {c_name:<22s} (N={len(eps):2d}): Acc = {m['exact_match_accuracy']:5.1f}%, Obj Acc = {m.get('slot_accuracies', {}).get('object', 0.0):5.1f}%")
        model_data["morphological_case_breakdown"] = case_metrics

        results_by_model[name] = model_data

    summary_output = {
        "experiment_id": "EXP-07",
        "description": "Syntactic, Morphological, and Pragmatic Robustness Evaluation",
        "num_episodes": len(test_episodes),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results_by_model,
    }

    out_file = output_dir / "robustness_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    print(f"\n[OK] Results saved to {out_file}")

    return summary_output


if __name__ == "__main__":
    run_robustness_test_experiment()
