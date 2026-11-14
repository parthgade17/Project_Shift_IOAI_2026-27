"""
Project SHIFT: Conversational Context Ablation Experiment.
IOAI 2027 Scientific Benchmark Experiment Suite (EXP-02).

Evaluates the necessity and sensitivity of conversational dialogue context.
Tests model performance across three conditions:
  1. Full Context: Standard multi-turn dialogue context intact.
  2. No Context (c=0): Context turns stripped completely.
  3. Shuffled Context: Context turns permuted in reverse order.
Measures performance drops on discourse phenomena (coreference, ellipsis, clarification, correction).
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
from shift.episodes import Episode
from shift.evaluator import Evaluator
from baselines.majority import MajorityBaseline
from baselines.surface_lookup import SurfaceLookupBaseline
from baselines.statistical import StatisticalLearner
from baselines.rule_based import RuleBasedLearner


def run_context_ablation_experiment(
    dataset_path: str = "datasets/hidden_test_full_reference.jsonl",
    output_dir: Path = None,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" [EXP-02] CONVERSATIONAL CONTEXT ABLATION EXPERIMENT")
    print(f" Source dataset: {dataset_path}")
    print("=" * 72)

    all_episodes = load_episodes_jsonl(dataset_path)

    # Filter or separate dialogue episodes
    dialogue_episodes = [
        ep for ep in all_episodes
        if ep.query.context and len(ep.query.context) > 0
    ]

    if not dialogue_episodes:
        # If no dialogue episodes in test split, use all episodes
        dialogue_episodes = all_episodes

    print(f"Total dialogue episodes with context: {len(dialogue_episodes)}")

    baselines = [
        ("Statistical Naive Bayes", StatisticalLearner()),
        ("Inductive Rule Learner", RuleBasedLearner()),
        ("Surface Lookup Baseline", SurfaceLookupBaseline()),
        ("Majority Baseline", MajorityBaseline()),
    ]

    # Condition 1: Full Context
    full_eps = dialogue_episodes
    gt = {ep.episode_id: ep.query_label for ep in full_eps}
    ep_dict = {ep.episode_id: ep for ep in full_eps}

    # Condition 2: No Context
    no_ctx_eps = []
    for ep in dialogue_episodes:
        c = copy.deepcopy(ep)
        c.query.context = []
        no_ctx_eps.append(c)

    # Condition 3: Shuffled Context (reverse turn order)
    shuf_ctx_eps = []
    for ep in dialogue_episodes:
        c = copy.deepcopy(ep)
        c.query.context = list(reversed(c.query.context))
        shuf_ctx_eps.append(c)

    conditions = [
        ("Full Context", full_eps),
        ("No Context (c=0)", no_ctx_eps),
        ("Shuffled Context (Reversed)", shuf_ctx_eps),
    ]

    results_by_model: Dict[str, Any] = {}

    for name, model in baselines:
        print(f"\n--> Model: {name}")
        model_results = {}
        for cond_name, eps in conditions:
            preds = {ep.episode_id: model.predict_episode(ep) for ep in eps}
            m = Evaluator.evaluate(preds, gt, ep_dict)
            model_results[cond_name] = {
                "exact_match_accuracy": m["exact_match_accuracy"],
                "macro_slot_f1": m["macro_slot_f1"],
                "reference_resolution_accuracy": m.get("reference_resolution_accuracy", 0.0),
            }
            print(f"    * {cond_name:<30}: Exact Match = {m['exact_match_accuracy']:5.1f}%, Ref Res = {m.get('reference_resolution_accuracy', 0.0):5.1f}%")

        full_acc = model_results["Full Context"]["exact_match_accuracy"]
        no_acc = model_results["No Context (c=0)"]["exact_match_accuracy"]
        shuf_acc = model_results["Shuffled Context (Reversed)"]["exact_match_accuracy"]

        model_results["ablation_drop_no_context"] = round(full_acc - no_acc, 2)
        model_results["ablation_drop_shuffled"] = round(full_acc - shuf_acc, 2)
        print(f"    --> Drop without Context : {model_results['ablation_drop_no_context']:+5.1f}%")
        print(f"    --> Drop with Shuffled  : {model_results['ablation_drop_shuffled']:+5.1f}%")

        results_by_model[name] = model_results

    # Breakdown by discourse phenomenon for Inductive Rule Learner
    learner = RuleBasedLearner()
    phenomena_breakdown = {}
    for phenom in ["coreference", "ellipsis", "clarification", "correction"]:
        p_eps = [ep for ep in dialogue_episodes if ep.query.phenomenon == phenom]
        if not p_eps:
            continue
        p_gt = {ep.episode_id: ep.query_label for ep in p_eps}
        p_dict = {ep.episode_id: ep for ep in p_eps}

        p_no_ctx = []
        for ep in p_eps:
            c = copy.deepcopy(ep)
            c.query.context = []
            p_no_ctx.append(c)

        preds_full = {ep.episode_id: learner.predict_episode(ep) for ep in p_eps}
        preds_no = {ep.episode_id: learner.predict_episode(ep) for ep in p_no_ctx}

        m_full = Evaluator.evaluate(preds_full, p_gt, p_dict)
        m_no = Evaluator.evaluate(preds_no, p_gt, p_dict)

        phenomena_breakdown[phenom] = {
            "count": len(p_eps),
            "full_context_accuracy": m_full["exact_match_accuracy"],
            "no_context_accuracy": m_no["exact_match_accuracy"],
            "drop": round(m_full["exact_match_accuracy"] - m_no["exact_match_accuracy"], 2),
        }

    summary_output = {
        "experiment_id": "EXP-02",
        "description": "Conversational Context Ablation (Full vs No-Context vs Shuffled)",
        "num_dialogue_episodes": len(dialogue_episodes),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results_by_model": results_by_model,
        "phenomena_breakdown_rule_learner": phenomena_breakdown,
    }

    out_file = output_dir / "context_ablation.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    print(f"\n[OK] Results saved to {out_file}")

    return summary_output


if __name__ == "__main__":
    run_context_ablation_experiment()
