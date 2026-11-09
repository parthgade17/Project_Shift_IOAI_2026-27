"""
Project SHIFT: Component Ablation Study.
Measures the marginal contribution of each structural component:
1. Full System (Unablated control)
2. No Community Metadata (Tests reliance on community ID)
3. No Historical Era Metadata (Tests reliance on temporal ID)
4. No Dialogue History (Context ablation for discourse tracking)
5. No Support Examples (k = 0 support ablation)
6. No Syntactic Alignment (Bag-of-words heuristic)
"""

from __future__ import annotations
import sys
import copy
import json
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shift.serialize import load_episodes_jsonl
from shift.evaluator import Evaluator
from baselines.rule_based import RuleBasedLearner


def run_ablation_study(
    dataset_path: str = "datasets/hidden_test_full_reference.jsonl",
    output_path: str = "results/ablation_study_results.json"
) -> Dict[str, Any]:
    print("=" * 70)
    print(" PROJECT SHIFT: Component Ablation Study")
    print("=" * 70)

    episodes = load_episodes_jsonl(dataset_path)
    gt = {ep.episode_id: ep.query_label for ep in episodes}
    ep_dict = {ep.episode_id: ep for ep in episodes}

    learner = RuleBasedLearner()

    ablations = [
        ("Full System (Unablated Control)", "none"),
        ("No Community Metadata", "no_community"),
        ("No Historical Era Metadata", "no_period"),
        ("No Conversational Context", "no_context"),
        ("No Support Examples (k=0)", "no_support"),
    ]

    results: Dict[str, Any] = {}

    print(f"\n{'Ablation Condition':<35} | {'Exact Match':<12} | {'Disentangle':<12} | {'Delta EM':<10}")
    print("-" * 75)

    control_em = 0.0

    for name, condition in ablations:
        ablated_eps = []
        for ep in episodes:
            c = copy.deepcopy(ep)
            if condition == "no_community":
                c.community_id = "unknown"
            elif condition == "no_period":
                c.period_id = "unknown"
            elif condition == "no_context":
                c.query.context = []
            elif condition == "no_support":
                c.support_set = []
            ablated_eps.append(c)

        preds = {ep.episode_id: learner.predict_episode(ep) for ep in ablated_eps}
        eval_res = Evaluator.evaluate(preds, gt, ep_dict)

        em = eval_res["exact_match_accuracy"]
        dis = eval_res["disentanglement_accuracy"]

        if condition == "none":
            control_em = em
            delta_str = " control"
        else:
            diff = em - control_em
            delta_str = f" {diff:+.2f}%"

        results[name] = {
            "condition": condition,
            "exact_match_accuracy": em,
            "disentanglement_accuracy": dis,
            "delta_from_control": round(em - control_em, 2)
        }

        print(f"{name:<35} | {em:10.2f}% | {dis:10.2f}% | {delta_str:<10}")

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[OK] Ablation study exported to {output_path}\n")
    return results


if __name__ == "__main__":
    run_ablation_study()
