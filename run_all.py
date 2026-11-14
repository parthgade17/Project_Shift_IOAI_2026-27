"""
Project SHIFT: Master End-to-End Orchestrator (Single-Command Execution).
IOAI 2027 Singapore Scientific Benchmark.

Executes the entire Project SHIFT pipeline in a single command:
1. Automated Unit Test Suite (75 automated tests)
2. Dataset Generation Pipeline & 6-Check Learnability Quality Gates
3. Benchmark Experiments, Error Diagnostics & Multi-Format Reports
4. Contestant Workflow Simulation & Official Jury Grading
5. Computational Profiling & IOAI Constraint Verification
6. Deterministic Contestant & Author Release Packaging (Zero-Leakage Verified)

Usage:
    python run_all.py
"""

from __future__ import annotations
import sys
import time
import unittest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_complete_project() -> int:
    t_start = time.perf_counter()
    print("\n" + "=" * 76)
    print(" PROJECT SHIFT: MASTER END-TO-END EXECUTION PIPELINE (IOAI 2027)")
    print("=" * 76)

    # -------------------------------------------------------------
    # Stage 1: Automated Unit Test Suite
    # -------------------------------------------------------------
    print("\n[Stage 1/6] Running Automated Unit Tests (75 Tests)...")
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=1)
    test_result = runner.run(suite)
    if not test_result.wasSuccessful():
        print("[FAIL] Unit tests failed. Aborting master execution.")
        return 1
    print(f"  [PASS] {test_result.testsRun} unit tests passed successfully.")

    # -------------------------------------------------------------
    # Stage 2: Dataset Generation Pipeline & Quality Gates
    # -------------------------------------------------------------
    print("\n[Stage 2/6] Running Dataset Generation & 6-Check Learnability Quality Gates...")
    from scripts.generate_datasets import main as generate_datasets_main
    try:
        generate_datasets_main()
        print("  [PASS] Datasets generated and quality gates verified.")
    except Exception as e:
        print(f"[FAIL] Dataset generation error: {e}")
        return 1

    # -------------------------------------------------------------
    # Stage 3: Benchmark Experiments & Diagnostics
    # -------------------------------------------------------------
    print("\n[Stage 3/6] Running Benchmark Experiments, Diagnostics & Multi-Format Reports...")
    from scripts.run_experiments import run_experiments
    from analysis.error_analysis import run_error_analysis
    from analysis.model_comparison import run_model_comparison
    try:
        run_experiments()
        run_error_analysis()
        run_model_comparison()
        print("  [PASS] Benchmark experiments and model comparisons completed.")
    except Exception as e:
        print(f"[FAIL] Benchmark experiments error: {e}")
        return 1

    # -------------------------------------------------------------
    # Stage 4: Contestant Workflow Simulation & Jury Grading
    # -------------------------------------------------------------
    print("\n[Stage 4/6] Simulating Contestant Workflow & Official Jury Grading...")
    from scripts.submission_test import simulate_contestant_workflow
    try:
        success = simulate_contestant_workflow()
        if not success:
            print("[FAIL] Contestant workflow simulation failed.")
            return 1
        print("  [PASS] Contestant workflow and jury grading validated.")
    except Exception as e:
        print(f"[FAIL] Contestant workflow error: {e}")
        return 1

    # -------------------------------------------------------------
    # Stage 5: Computational Profiling & IOAI Compliance
    # -------------------------------------------------------------
    print("\n[Stage 5/6] Profiling Runtime & Memory against IOAI Resource Constraints...")
    from scripts.profile_ioai_limits import profile_ioai_limits
    try:
        profile = profile_ioai_limits()
        if not profile["all_passed"]:
            print("[FAIL] Non-compliant resource usage detected.")
            return 1
        print("  [PASS] Resource profiler passed all IOAI constraints.")
    except Exception as e:
        print(f"[FAIL] Profiling error: {e}")
        return 1

    # -------------------------------------------------------------
    # Stage 6: Deterministic Official Release Packaging
    # -------------------------------------------------------------
    print("\n[Stage 6/6] Generating Deterministic Contestant and Author Release Packages...")
    from scripts.package_release import package_releases
    try:
        c_zip, a_zip = package_releases()
        print(f"  [PASS] Release archives built: {c_zip.name}, {a_zip.name}")
    except Exception as e:
        print(f"[FAIL] Release packaging error: {e}")
        return 1

    total_elapsed = time.perf_counter() - t_start

    # -------------------------------------------------------------
    # Final Executive Summary
    # -------------------------------------------------------------
    rule_learner_prof = profile["model_profiles"]["Inductive Rule Learner"]
    print("\n" + "=" * 76)
    print(" PROJECT SHIFT: ALL 6 PIPELINE STAGES COMPLETED SUCCESSFULLY")
    print("=" * 76)
    print(f"  * Total Execution Time : {total_elapsed:.2f} seconds")
    print(f"  * Unit Test Suite      : {test_result.testsRun} / {test_result.testsRun} Passed (100%)")
    print(f"  * Memory Peak          : {rule_learner_prof['peak_memory_mb']:.2f} MB (IOAI Limit: < 2048 MB)")
    print(f"  * Inference Latency    : {rule_learner_prof['latency_ms_per_episode']:.3f} ms/ep (IOAI Limit: < 2000 ms)")
    print(f"  * External Dependencies: ZERO (Pure Python 3.10+ Standard Library)")
    print(f"  * Official Releases    : releases/contestant_package_v1.0.zip")
    print(f"                           releases/author_package_v1.0.zip")
    print(f"  * Reports Generated    : results/shift_benchmark_report.md (.tex, .json)")
    print(f"                           results/model_comparison.md (.tex, .json)")
    print(f"                           results/error_analysis.json")
    print(f"                           reports/RESEARCH_EXPERIMENT_LOG.md")
    print("=" * 76 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(run_complete_project())
