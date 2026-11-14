"""
Project SHIFT: Compositional Generalization Experiment.
IOAI 2027 Scientific Benchmark Experiment Suite (EXP-05).

Evaluates systematic compositional generalization on held-out action-object compositions (Level 5).
Compares performance on in-distribution test episodes (test_iid.jsonl) against out-of-distribution
action-object pairs (test_ood.jsonl). Measures whether models can systematically recombine familiar
lexical primitives into novel semantic frames without degradation.
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


def run_composition_test_experiment(
    iid_path: str = "datasets/test_iid.jsonl",
    ood_path: str = "datasets/test_ood.jsonl",
    output_dir: Path = None,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" [EXP-05] COMPOSITIONAL GENERALIZATION EXPERIMENT (LEVEL 5)")
    print(f" In-distribution split (IID) : {iid_path}")
    print(f" Out-of-distribution (OOD)   : {ood_path}")
    print("=" * 72)

    iid_episodes = load_episodes_jsonl(iid_path)
    ood_episodes = load_episodes_jsonl(ood_path)

    iid_gt = {ep.episode_id: ep.query_label for ep in iid_episodes}
    iid_dict = {ep.episode_id: ep for ep in iid_episodes}

    ood_gt = {ep.episode_id: ep.query_label for ep in ood_episodes}
    ood_dict = {ep.episode_id: ep for ep in ood_episodes}

    baselines = [
        ("Majority Baseline", MajorityBaseline()),
        ("Surface Lookup Baseline", SurfaceLookupBaseline()),
        ("Statistical Naive Bayes", StatisticalLearner()),
        ("Inductive Rule Learner", RuleBasedLearner()),
    ]

    results: Dict[str, Any] = {}

    for name, model in baselines:
        print(f"\n--> Model: {name}")
        preds_iid = {ep.episode_id: model.predict_episode(ep) for ep in iid_episodes}
        m_iid = Evaluator.evaluate(preds_iid, iid_gt, iid_dict)

        preds_ood = {ep.episode_id: model.predict_episode(ep) for ep in ood_episodes}
        m_ood = Evaluator.evaluate(preds_ood, ood_gt, ood_dict)

        acc_iid = m_iid["exact_match_accuracy"]
        acc_ood = m_ood["exact_match_accuracy"]
        comp_gap = round(acc_iid - acc_ood, 2)

        # Measure action-object compound accuracy on OOD
        compound_correct = 0
        for ep in ood_episodes:
            p = preds_ood[ep.episode_id]
            g = ood_gt[ep.episode_id]
            if p.action == g.action and p.object == g.object:
                compound_correct += 1
        compound_acc = round(compound_correct / len(ood_episodes) * 100.0, 2)

        results[name] = {
            "iid": {
                "exact_match_accuracy": acc_iid,
                "macro_slot_f1": m_iid["macro_slot_f1"],
                "action_accuracy": m_iid.get("slot_accuracies", {}).get("action", 0.0),
                "object_accuracy": m_iid.get("slot_accuracies", {}).get("object", 0.0),
            },
            "ood_compositional": {
                "exact_match_accuracy": acc_ood,
                "macro_slot_f1": m_ood["macro_slot_f1"],
                "action_accuracy": m_ood.get("slot_accuracies", {}).get("action", 0.0),
                "object_accuracy": m_ood.get("slot_accuracies", {}).get("object", 0.0),
                "action_object_compound_accuracy": compound_acc,
            },
            "compositional_gap": comp_gap,
        }

        print(f"    * IID Exact Match         : {acc_iid:5.1f}%")
        print(f"    * OOD Compositional Acc   : {acc_ood:5.1f}%")
        print(f"    * Compound (Action+Obj)   : {compound_acc:5.1f}%")
        print(f"    * Compositional Gap (Delta): {comp_gap:+5.1f}%")

    summary_output = {
        "experiment_id": "EXP-05",
        "description": "Systematic Compositional Generalization on Held-out Action-Object Combinations",
        "num_iid_episodes": len(iid_episodes),
        "num_ood_episodes": len(ood_episodes),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }

    out_file = output_dir / "compositional_ood.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    print(f"\n[OK] Results saved to {out_file}")

    return summary_output


if __name__ == "__main__":
    run_composition_test_experiment()
