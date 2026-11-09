"""
Project SHIFT: Master End-to-End Orchestrator (Single-Command Execution).
IOAI 2027 Scientific Benchmark.

Executes the entire Project SHIFT pipeline in a single command:
1. Unit Test Suite (48 automated tests)
2. Dataset Generation Pipeline & Learnability Quality Gates
3. Benchmark Experiments & Multi-Format Report Generation
4. Contestant Workflow Simulation & Official Jury Grading
5. Computational Profiling & IOAI Constraint Verification

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
    # Step 1: Automated Unit Test Suite
    # -------------------------------------------------------------
    print("\n[Stage 1/5] Running Automated Unit Tests (48 Tests)...")
    suite = unittest.defaultTestLoader.discover(str(PROJECT_ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=1)
    test_result = runner.run(suite)
    if not test_result.wasSuccessful():
        print("[FAIL] Unit tests failed. Aborting master execution.")
        return 1
    print(f"  [PASS] {test_result.testsRun} unit tests passed successfully.")

    # -------------------------------------------------------------
    # Step 2: Dataset Generation Pipeline & Quality Gates
    # -------------------------------------------------------------
    print("\n[Stage 2/5] Running Dataset Generation & Quality Audits...")
    from scripts.generate_datasets import main as generate_datasets_main
    try:
        generate_datasets_main()
        print("  [PASS] Datasets generated and quality gates verified.")
    except Exception as e:
        print(f"[FAIL] Dataset generation error: {e}")
        return 1

    # -------------------------------------------------------------
    # Step 3: Benchmark Experiments & Multi-Format Reports
    # -------------------------------------------------------------
    print("\n[Stage 3/5] Running Official Benchmark Experiments & Generating Reports...")
    from scripts.run_experiments import run_experiments
    try:
        run_experiments()
        print("  [PASS] Benchmark experiments completed and reports exported.")
    except Exception as e:
        print(f"[FAIL] Benchmark experiments error: {e}")
        return 1

    # -------------------------------------------------------------
    # Step 4: Contestant Workflow Simulation & Jury Grading
    # -------------------------------------------------------------
    print("\n[Stage 4/5] Simulating Contestant Workflow & Official Jury Grading...")
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
    # Step 5: Computational Profiling & IOAI Compliance
    # -------------------------------------------------------------
    print("\n[Stage 5/5] Profiling Runtime & Memory against IOAI Limits...")
    from scripts.runtime_profiler import run_runtime_profiler
    try:
        profile = run_runtime_profiler()
        print("  [PASS] Runtime profiler completed.")
    except Exception as e:
        print(f"[FAIL] Profiling error: {e}")
        return 1

    total_elapsed = time.perf_counter() - t_start

    # -------------------------------------------------------------
    # Final Executive Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 76)
    print(" PROJECT SHIFT: ALL STAGES COMPLETED SUCCESSFULLY")
    print("=" * 76)
    print(f"  * Total Execution Time : {total_elapsed:.2f} seconds")
    print(f"  * Unit Test Suite      : {test_result.testsRun} / {test_result.testsRun} Passed (100%)")
    print(f"  * Memory Footprint     : {profile['memory_peak_mb']:.2f} MB (IOAI Limit: < 2048 MB)")
    print(f"  * Inference Latency    : {profile['latency_ms_per_episode']:.3f} ms/ep (IOAI Limit: < 100 ms)")
    print(f"  * External Dependencies: ZERO (Pure Python 3.10+ Standard Library)")
    print(f"  * Formatted Reports    : results/shift_benchmark_report.md (.tex, .json)")
    print("=" * 76 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(run_complete_project())
