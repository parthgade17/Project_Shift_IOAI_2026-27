"""
Project SHIFT: Systematic Generalisation Experiment.
Evaluates model robustness across multiple generalization dimensions:
1. Seen IID (In-Distribution baseline)
2. Compositional OOD (Unseen action-object semantic combinations)
3. Held-out Community (Unseen dialect and lexical shifts)
4. Held-out Historical Era (Unseen temporal phonology and affixes)
5. Counterfactual Pairs (Contextual sensitivity on identical surface queries)
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shift.serialize import load_episodes_jsonl
from shift.evaluator import Evaluator
from baselines.majority import MajorityBaseline
from baselines.statistical import StatisticalLearner
from baselines.rule_based import RuleBasedLearner


def run_generalisation_test(
    output_path: str = "results/generalisation_test_results.json"
) -> Dict[str, Any]:
    print("=" * 70)
    print(" PROJECT SHIFT: Systematic Generalisation Experiment")
    print("=" * 70)

    splits = [
        ("Seen IID (In-Distribution)", "datasets/test_iid.jsonl"),
        ("Compositional OOD (Unseen Combos)", "datasets/test_ood.jsonl"),
        ("Held-out Community (Unseen Dialect)", "datasets/test_heldout_community.jsonl"),
        ("Held-out Period (Unseen Epoch)", "datasets/test_heldout_period.jsonl"),
        ("Counterfactual Pairs (Context Bias)", "datasets/test_counterfactual.jsonl"),
    ]

    models = [
        ("Inductive Rule Learner", RuleBasedLearner()),
        ("Statistical Naive Bayes", StatisticalLearner()),
        ("Majority Baseline", MajorityBaseline()),
    ]

    all_results: Dict[str, Any] = {"splits": {}, "summary_table": {}}

    print(f"\n{'Split Name':<38} | {'Episodes':<8} | {'Rule Learner':<14} | {'Naive Bayes':<14} | {'Majority':<10}")
    print("-" * 92)

    for split_name, split_path in splits:
        p = Path(split_path)
        if not p.exists():
            print(f"[SKIP] {split_path} not found.")
            continue

        episodes = load_episodes_jsonl(p)
        gt = {ep.episode_id: ep.query_label for ep in episodes}
        ep_dict = {ep.episode_id: ep for ep in episodes}

        split_res = {}
        row_str = f"{split_name:<38} | {len(episodes):<8} |"

        for m_name, model in models:
            preds = {ep.episode_id: model.predict_episode(ep) for ep in episodes}
            eval_res = Evaluator.evaluate(preds, gt, ep_dict)
            acc = eval_res["exact_match_accuracy"]
            split_res[m_name] = eval_res
            row_str += f" {acc:12.2f}% |"

        print(row_str)
        all_results["splits"][split_name] = split_res

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[OK] Results exported to {output_path}\n")
    return all_results


if __name__ == "__main__":
    run_generalisation_test()
