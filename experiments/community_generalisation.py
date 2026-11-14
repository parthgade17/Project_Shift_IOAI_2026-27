"""
Project SHIFT: Community Generalization Experiment.
IOAI 2027 Scientific Benchmark Experiment Suite (EXP-03).

Evaluates generalization to unseen community dialects (held-out community test set).
Compares performance on seen communities (test_iid.jsonl) against an unseen community (test_heldout_community.jsonl).
Tests both zero-shot and few-shot in-context adaptation.
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


def run_community_generalisation_experiment(
    iid_path: str = "datasets/test_iid.jsonl",
    heldout_path: str = "datasets/test_heldout_community.jsonl",
    output_dir: Path = None,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" [EXP-03] COMMUNITY GENERALIZATION EXPERIMENT")
    print(f" Seen communities split   : {iid_path}")
    print(f" Held-out community split : {heldout_path}")
    print("=" * 72)

    iid_episodes = load_episodes_jsonl(iid_path)
    heldout_episodes = load_episodes_jsonl(heldout_path)

    iid_gt = {ep.episode_id: ep.query_label for ep in iid_episodes}
    iid_dict = {ep.episode_id: ep for ep in iid_episodes}

    heldout_gt = {ep.episode_id: ep.query_label for ep in heldout_episodes}
    heldout_dict = {ep.episode_id: ep for ep in heldout_episodes}

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

        preds_heldout = {ep.episode_id: model.predict_episode(ep) for ep in heldout_episodes}
        m_heldout = Evaluator.evaluate(preds_heldout, heldout_gt, heldout_dict)

        acc_iid = m_iid["exact_match_accuracy"]
        acc_heldout = m_heldout["exact_match_accuracy"]
        gen_gap = round(acc_iid - acc_heldout, 2)

        results[name] = {
            "seen_iid": {
                "exact_match_accuracy": acc_iid,
                "macro_slot_f1": m_iid["macro_slot_f1"],
                "disentanglement_accuracy": m_iid.get("disentanglement_accuracy", 0.0),
            },
            "unseen_heldout_community": {
                "exact_match_accuracy": acc_heldout,
                "macro_slot_f1": m_heldout["macro_slot_f1"],
                "disentanglement_accuracy": m_heldout.get("disentanglement_accuracy", 0.0),
            },
            "generalization_gap": gen_gap,
        }

        print(f"    * Seen IID Accuracy       : {acc_iid:5.1f}%")
        print(f"    * Held-out Dialect Acc    : {acc_heldout:5.1f}%")
        print(f"    * Generalization Gap (Delta): {gen_gap:+5.1f}%")

    summary_output = {
        "experiment_id": "EXP-03",
        "description": "Generalization to Unseen Community Dialects (Seen vs Held-out)",
        "num_iid_episodes": len(iid_episodes),
        "num_heldout_episodes": len(heldout_episodes),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": results,
    }

    out_file = output_dir / "community_generalisation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    print(f"\n[OK] Results saved to {out_file}")

    return summary_output


if __name__ == "__main__":
    run_community_generalisation_experiment()
