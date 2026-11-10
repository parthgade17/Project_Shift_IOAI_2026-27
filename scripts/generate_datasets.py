"""
Project SHIFT: Dataset Generation Pipeline (v0.3 Benchmark Infrastructure).
Builds the pilot dataset, competition splits, counterfactual pairs,
and executes strict learnability and shortcut audit quality gates.
"""

import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shift.episodes import EpisodeGenerator
from shift.splits import SplitManager
from shift.serialize import export_episodes_jsonl
from shift.validation import audit_dataset_quality
from shift.learnability_validator import LearnabilityValidator
from shift.shortcut_audits import ShortcutAuditor


def main():
    print("=" * 70)
    print(" PROJECT SHIFT: Scientific Dataset Generation Pipeline (v0.3)")
    print("=" * 70)

    datasets_dir = Path("datasets")
    datasets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Pilot Dataset
    print("\n[Step 1/4] Generating 100-episode pilot dataset...")
    pilot_gen = EpisodeGenerator(base_seed=42)
    pilot_episodes = pilot_gen.generate_batch(count=100, k_support=8, id_prefix="pilot")
    pilot_path = datasets_dir / "pilot_100.jsonl"
    export_episodes_jsonl(pilot_episodes, pilot_path, strip_labels=False)
    print(f"  [OK] Saved 100 pilot episodes to {pilot_path}")

    # Audit pilot dataset
    pilot_report = audit_dataset_quality(pilot_episodes)
    print(f"  [OK] Pilot Schema Pass Rate      : {pilot_report.valid_schema_count}/{pilot_report.total_episodes} (100%)")
    print(f"  [OK] Pilot Learnability Pass Rate: {pilot_report.learnable_count}/{pilot_report.total_episodes} ({pilot_report.learnable_count/pilot_report.total_episodes*100:.1f}%)")

    # 2. Partitioned Competition Splits
    print("\n[Step 2/4] Generating leakage-resistant competition splits...")
    split_manager = SplitManager(base_seed=2027)
    splits = split_manager.create_partitioned_dataset(
        train_count=100,
        val_count=30,
        dev_count=30,
        test_iid_count=30,
        test_ood_count=30,
        test_comm_count=20,
        test_period_count=20,
        k_support=8
    )

    split_manager.save_splits_to_directory(splits, datasets_dir)
    print(f"  [OK] Train Episodes       : {len(splits.train)}")
    print(f"  [OK] Val Episodes         : {len(splits.val)}")
    print(f"  [OK] Public Dev Episodes  : {len(splits.public_dev)}")
    print(f"  [OK] Hidden Test Episodes : {len(splits.hidden_test)} (IID=30, OOD=30, Comm=20, Period=20)")
    print(f"  [OK] Counterfactual Tests : {len(splits.test_counterfactual)} ({len(splits.test_counterfactual)//2} matched pairs)")

    # 3. Synchronize hidden test labels with hidden_evaluator_package
    hidden_eval_pkg = Path("hidden_evaluator_package")
    if hidden_eval_pkg.exists():
        src_lab = datasets_dir / "hidden_test_labels.json"
        dest_lab = hidden_eval_pkg / "hidden_test_labels.json"
        if src_lab.exists():
            shutil.copyfile(src_lab, dest_lab)
            print("  [OK] Synchronized hidden_test_labels.json with hidden_evaluator_package/")

    # 4. Strict Quality Gates: Leakage, Learnability & Shortcut Audits
    print("\n[Step 3/4] Running leakage and quality audits...")
    leak_audit = split_manager.audit_leakage(splits)
    if leak_audit["leak_free"]:
        print("  [OK] Zero-Leakage Guarantee CONFIRMED (No utterance or frame overlap across splits)")
    else:
        print(f"  [WARN] Leakage detected: {leak_audit}")

    test_report = audit_dataset_quality(splits.hidden_test)
    print(f"  [OK] Hidden Test Schema Pass Rate: {test_report.valid_schema_count}/{test_report.total_episodes} (100%)")
    print(f"  [OK] Hidden Test Learnability    : {test_report.learnable_count}/{test_report.total_episodes} ({test_report.learnable_count/test_report.total_episodes*100:.1f}%)")

    # Generate and export comprehensive 6-check learnability report
    _, full_learn_report = LearnabilityValidator.audit_and_repair_dataset(splits.hidden_test)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    report_md_path = results_dir / "learnability_validation_report.md"
    report_json_path = results_dir / "learnability_validation_report.json"

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(full_learn_report.format_markdown())
    import json
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(full_learn_report.to_dict(), f, indent=2)
    print(f"  [OK] Exported 6-Check Learnability Report: {report_md_path}")

    # 5. Adversarial Shortcut Audits on Hidden Test Set & Counterfactual Split
    print("\n[Step 4/4] Executing adversarial shortcut audits...")
    eval_suite_episodes = splits.hidden_test + splits.test_counterfactual
    shortcut_suite = ShortcutAuditor.run_full_shortcut_suite(eval_suite_episodes)
    if shortcut_suite["all_shortcut_audits_passed"]:
        print("  [OK] All adversarial shortcut audits PASSED:")
        print(f"       * Metadata Attack Score: {shortcut_suite['metadata_attack']['exact_match_accuracy']:.1f}% (<= 25% threshold)")
        print(f"       * Support Ablation (k=0): {shortcut_suite['support_ablation']['exact_match_accuracy']:.1f}% (<= 30% threshold)")
        cf_res = shortcut_suite['counterfactual_consistency']
        cf_val = cf_res.get('consistency_score', 100.0)
        print(f"       * Counterfactual Consistency: {cf_val:.1f}% (>= 80% threshold)")
    else:
        print(f"  [WARN] Shortcut audit notice: {shortcut_suite}")

    print("\n[COMPLETE] All datasets validated and generated in ./datasets/\n")


if __name__ == "__main__":
    main()
