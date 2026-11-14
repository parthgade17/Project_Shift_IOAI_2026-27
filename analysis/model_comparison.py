"""
Project SHIFT: Cross-Model Comparative Benchmark Analysis.
IOAI 2027 Scientific Benchmark Analysis Suite.

Generates comprehensive comparison matrices across all baseline paradigms:
  1. Majority Baseline (trivial empirical prior)
  2. Surface Lookup (nearest-neighbour token overlap)
  3. Statistical Naive Bayes (probabilistic MLE n-gram alignment)
  4. Inductive Rule Learner (symbolic few-shot grammar and discourse parser)

Computes Exact Match, Slot F1, Disentanglement, Discourse Reference Resolution,
Counterfactual Consistency, Inference Latency, and Composite IOAI Score.
Outputs formatted Markdown, LaTeX tables, and JSON.
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


def run_model_comparison(
    test_path: str = "datasets/hidden_test_full_reference.jsonl",
    output_dir: Path = None,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" PROJECT SHIFT: CROSS-MODEL COMPARATIVE BENCHMARK ANALYSIS")
    print(f" Reference test set: {test_path}")
    print("=" * 72)

    test_episodes = load_episodes_jsonl(test_path)
    gt = {ep.episode_id: ep.query_label for ep in test_episodes}
    ep_dict = {ep.episode_id: ep for ep in test_episodes}

    baselines = [
        ("Majority Baseline", MajorityBaseline()),
        ("Surface Lookup Baseline", SurfaceLookupBaseline()),
        ("Statistical Naive Bayes", StatisticalLearner()),
        ("Inductive Rule Learner", RuleBasedLearner()),
    ]

    metrics_table: Dict[str, Any] = {}

    for name, model in baselines:
        t0 = time.perf_counter()
        preds = {ep.episode_id: model.predict_episode(ep) for ep in test_episodes}
        elapsed = time.perf_counter() - t0
        latency_ms = (elapsed / len(test_episodes)) * 1000.0

        m = Evaluator.evaluate(preds, gt, ep_dict)
        m["avg_latency_ms"] = round(latency_ms, 3)

        metrics_table[name] = m

    # Generate Markdown Table
    md_lines = [
        "# Project SHIFT: Comparative Benchmark Results",
        "",
        "| Model Architecture | Exact Match (%) | Macro Slot F1 (%) | Disentanglement (%) | Reference Res (%) | Counterfactual (%) | Composite Score (/100) | Latency (ms/ep) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for name, m in metrics_table.items():
        em = m["exact_match_accuracy"]
        f1 = m["macro_slot_f1"]
        dis = m.get("disentanglement_accuracy", 0.0)
        ref = m.get("reference_resolution_accuracy", 0.0)
        cf = m.get("counterfactual_consistency", {}).get("consistency_score", 0.0)
        comp = m.get("composite_competition_score", 0.0)
        lat = m.get("avg_latency_ms", 0.0)

        md_lines.append(
            f"| **{name}** | {em:.1f}% | {f1:.1f}% | {dis:.1f}% | {ref:.1f}% | {cf:.1f}% | **{comp:.1f}** | {lat:.3f} |"
        )

    # Generate LaTeX Table
    latex_lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Official Benchmark Evaluation on Project SHIFT Hidden Test Set (IOAI 2027 Singapore).}",
        r"\label{tab:shift_benchmark_results}",
        r"\begin{tabular}{lccccccr}",
        r"\toprule",
        r"\textbf{Model Paradigm} & \textbf{Exact Match} & \textbf{Slot F1} & \textbf{Disentangle} & \textbf{Ref.~Res.} & \textbf{Counterfac.} & \textbf{Composite} & \textbf{Latency} \\",
        r"\midrule",
    ]

    for name, m in metrics_table.items():
        em = m["exact_match_accuracy"]
        f1 = m["macro_slot_f1"]
        dis = m.get("disentanglement_accuracy", 0.0)
        ref = m.get("reference_resolution_accuracy", 0.0)
        cf = m.get("counterfactual_consistency", {}).get("consistency_score", 0.0)
        comp = m.get("composite_competition_score", 0.0)
        lat = m.get("avg_latency_ms", 0.0)

        bold_open = r"\textbf{" if "Rule" in name else ""
        bold_close = "}" if "Rule" in name else ""
        latex_lines.append(
            f"{bold_open}{name}{bold_close} & {em:.1f}\\% & {f1:.1f}\\% & {dis:.1f}\\% & {ref:.1f}\\% & {cf:.1f}\\% & {bold_open}{comp:.1f}{bold_close} & {lat:.3f}ms \\\\"
        )

    latex_lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])

    markdown_str = "\n".join(md_lines)
    latex_str = "\n".join(latex_lines)

    print("\n" + markdown_str + "\n")

    summary_output = {
        "analysis_type": "cross_model_comparison",
        "benchmark": "Project SHIFT (IOAI 2027)",
        "num_episodes": len(test_episodes),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": metrics_table,
        "markdown_table": markdown_str,
        "latex_table": latex_str,
    }

    out_json = output_dir / "model_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)

    out_md = output_dir / "model_comparison.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(markdown_str + "\n")

    out_tex = output_dir / "model_comparison.tex"
    with open(out_tex, "w", encoding="utf-8") as f:
        f.write(latex_str + "\n")

    print(f"[OK] Comparison exported to {out_json}, {out_md}, {out_tex}")

    return summary_output


if __name__ == "__main__":
    run_model_comparison()
