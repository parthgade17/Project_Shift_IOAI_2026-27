"""
Project SHIFT: Benchmark Experiment Runner & Report Generator.
Executes comprehensive benchmark evaluations: baseline comparisons, diagnostic
sub-splits, few-shot adaptation curves, model error taxonomy, and shortcut audits.
Outputs Markdown, LaTeX, and JSON reports via shift.evaluation_report.ReportGenerator.
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shift.episodes import EpisodeGenerator
from shift.serialize import load_episodes_jsonl
from shift.evaluator import Evaluator
from shift.diagnostics import ModelDiagnostics
from shift.shortcut_audits import ShortcutAuditor, GenuineMetadataAttackBaseline
from shift.evaluation_report import ReportGenerator
from baselines.majority import MajorityBaseline
from baselines.surface_lookup import SurfaceLookupBaseline
from baselines.statistical import StatisticalLearner
from baselines.rule_based import RuleBasedLearner


def run_experiments():
    print("=" * 70)
    print(" PROJECT SHIFT: Benchmark Experiments (v0.3 Infrastructure)")
    print("=" * 70)

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load hidden test reference episodes and ground truth
    test_episodes = load_episodes_jsonl("datasets/hidden_test_full_reference.jsonl")
    gt = {ep.episode_id: ep.query_label for ep in test_episodes}
    ep_dict = {ep.episode_id: ep for ep in test_episodes}

    report_gen = ReportGenerator(title="Project SHIFT: Official Benchmark Results (IOAI 2027)")

    # 1. Baseline Systems Evaluation on Hidden Test Set
    print("\n--- [Experiment 1] Baseline Evaluation on Hidden Test Set (N=100) ---")
    baselines = [
        ("Majority Baseline", MajorityBaseline()),
        ("Surface Lookup Baseline", SurfaceLookupBaseline()),
        ("Statistical Naive Bayes", StatisticalLearner()),
        ("Inductive Rule Learner", RuleBasedLearner()),
    ]

    baseline_metrics = {}
    for name, model in baselines:
        start_t = time.perf_counter()
        preds = {ep.episode_id: model.predict_episode(ep) for ep in test_episodes}
        latency = (time.perf_counter() - start_t) / len(test_episodes) * 1000.0

        metrics = Evaluator.evaluate(
            predictions=preds,
            ground_truth=gt,
            episodes_by_id=ep_dict
        )
        metrics["avg_latency_ms"] = round(latency, 3)
        baseline_metrics[name] = metrics

        # Model diagnostics
        diag_report = ModelDiagnostics.diagnose_all(test_episodes, preds)
        report_gen.add_model_results(name, metrics, diag_report)

        print(f"\n{name}:")
        print(f"  Exact Match Accuracy   : {metrics['exact_match_accuracy']}%")
        print(f"  IOAI Composite Score   : {metrics['composite_competition_score']} / 100")
        print(f"  Disentanglement Acc    : {metrics['disentanglement_accuracy']}%")
        print(f"  Reference Resolution   : {metrics['reference_resolution_accuracy']}%")
        print(f"  Avg Latency per Episode: {metrics['avg_latency_ms']} ms")

    # 2. Adversarial Shortcut Audits on Baselines
    print("\n--- [Experiment 2] Shortcut Audits across Baselines ---")
    audit_results = {}
    for name, model in baselines:
        clean_acc = baseline_metrics[name]["exact_match_accuracy"]

        # Support ablation
        supp_ablated = []
        for ep in test_episodes:
            import copy
            c = copy.deepcopy(ep)
            c.support_set = []
            supp_ablated.append(c)
        k0_preds = {ep.episode_id: model.predict_episode(ep) for ep in supp_ablated}
        k0_eval = Evaluator.evaluate(k0_preds, gt, ep_dict)
        k0_acc = k0_eval["exact_match_accuracy"]

        # Genuine Metadata attack
        meta_model = GenuineMetadataAttackBaseline(test_episodes)
        meta_preds = {ep.episode_id: meta_model.predict_episode(ep) for ep in test_episodes}
        meta_eval = Evaluator.evaluate(meta_preds, gt, ep_dict)
        meta_acc = meta_eval["exact_match_accuracy"]

        # Context ablation
        coref_eps = [ep for ep in test_episodes if ep.query.phenomenon in ("coreference", "ellipsis")]
        coref_gt = {ep.episode_id: ep.query_label for ep in coref_eps}
        coref_ablated = []
        for ep in coref_eps:
            import copy
            c = copy.deepcopy(ep)
            c.query.context = []
            coref_ablated.append(c)
        ctx_preds = {ep.episode_id: model.predict_episode(ep) for ep in coref_ablated}
        ctx_eval = Evaluator.evaluate(ctx_preds, coref_gt, {ep.episode_id: ep for ep in coref_eps})
        ctx_acc = ctx_eval["exact_match_accuracy"]

        audit_data = {
            "clean_accuracy": clean_acc,
            "support_ablation_accuracy": k0_acc,
            "context_ablation_accuracy": ctx_acc,
            "metadata_only_accuracy": meta_acc,
            "permutation_drop": round(clean_acc - k0_acc, 2)
        }
        audit_results[name] = audit_data
        report_gen.add_audit_results(name, audit_data)
        print(f"  * {name:<28}: Clean={clean_acc}%, k=0 Ablation={k0_acc}%, Meta Attack={meta_acc}%")

    # 3. Generalization Sub-Split Breakdown (Inductive Rule Learner)
    print("\n--- [Experiment 3] Diagnostic Sub-Split Analysis ---")
    sub_splits = [
        ("IID Test Set (Seen Communities & Eras)", "datasets/test_iid.jsonl"),
        ("Compositional OOD (Unseen Combos)", "datasets/test_ood.jsonl"),
        ("Held-out Community (Unseen Dialect)", "datasets/test_heldout_community.jsonl"),
        ("Held-out Period (Unseen Historical Epoch)", "datasets/test_heldout_period.jsonl"),
        ("Counterfactual Pairs (Context Sensitivity)", "datasets/test_counterfactual.jsonl"),
    ]

    sub_split_results = {}
    learner = RuleBasedLearner()
    for split_name, split_path in sub_splits:
        sub_eps = load_episodes_jsonl(split_path)
        sub_gt = {ep.episode_id: ep.query_label for ep in sub_eps}
        sub_dict = {ep.episode_id: ep for ep in sub_eps}
        preds = {ep.episode_id: learner.predict_episode(ep) for ep in sub_eps}
        m = Evaluator.evaluate(preds, sub_gt, sub_dict)
        sub_split_results[split_name] = m
        print(f"  * {split_name:<46}: Exact Match = {m['exact_match_accuracy']}% (N={len(sub_eps)})")
        if m.get("counterfactual_consistency", {}).get("total_pairs", 0) > 0:
            cf = m["counterfactual_consistency"]
            print(f"    --> Counterfactual Consistency: {cf['consistency_score']}% (Joint={cf.get('joint_accuracy', 0)}%, Sensitivity={cf.get('perturbation_sensitivity', 0)}%, N={cf['total_pairs']} pairs)")

    # 4. Few-Shot Adaptation Curve across Support Sizes k in {0, 4, 8, 16, 32} with Multiple Random Seeds
    print("\n--- [Experiment 4] Few-Shot Adaptation Curve (Multi-Seed: 42, 101, 2024) ---")
    k_values = [0, 4, 8, 16, 32]
    seeds = [42, 101, 2024]
    adaptation_curve = {}

    for k in k_values:
        seed_runs = {name: [] for name, _ in baselines}
        for s in seeds:
            curve_gen = EpisodeGenerator(base_seed=s)
            eval_batch = curve_gen.generate_batch(count=20, k_support=k, id_prefix=f"adapt_s{s}_k{k}")
            curve_gt = {ep.episode_id: ep.query_label for ep in eval_batch}
            curve_dict = {ep.episode_id: ep for ep in eval_batch}

            for name, model in baselines:
                preds = {ep.episode_id: model.predict_episode(ep) for ep in eval_batch}
                m = Evaluator.evaluate(preds, curve_gt, curve_dict)
                seed_runs[name].append(m["exact_match_accuracy"])

        # Compute mean across seeds
        k_results = {name: round(sum(vals) / len(vals), 2) for name, vals in seed_runs.items()}
        adaptation_curve[f"k={k}"] = k_results
        print(f"  * k={k:2d}: Inductive Learner={k_results['Inductive Rule Learner']:5.1f}%, Naive Bayes={k_results['Statistical Naive Bayes']:5.1f}%, Surface Lookup={k_results['Surface Lookup Baseline']:5.1f}%")

    # 5. Export Publication Reports
    print("\n--- [Exporting Reports] ---")
    exported_paths = report_gen.export_all(results_dir, base_name="shift_benchmark_report")
    print(f"  [OK] Markdown Report : {exported_paths['markdown']}")
    print(f"  [OK] LaTeX Table     : {exported_paths['latex']}")
    print(f"  [OK] JSON Benchmark  : {exported_paths['json']}")

    # Save complete experiment results JSON
    full_output = {
        "metadata": {
            "benchmark_version": "v0.3.0",
            "hidden_test_size": len(test_episodes),
            "generated_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "baseline_comparison": baseline_metrics,
        "shortcut_audits": audit_results,
        "sub_split_generalization": sub_split_results,
        "few_shot_adaptation_curve": adaptation_curve,
    }

    results_file = results_dir / "experiment_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)
    print(f"  [OK] Full Experiments: {results_file}")

    print("\n" + "=" * 70)
    print(" PUBLICATION BENCHMARK SUMMARY (MARKDOWN PREVIEW)")
    print("=" * 70)
    print(report_gen.generate_markdown())
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_experiments()
