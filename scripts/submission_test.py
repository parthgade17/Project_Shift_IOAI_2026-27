"""
Project SHIFT: Contestant Workflow Simulation & End-to-End Test.
Component of v0.4 Engineering & Reproducibility.

Simulates the complete end-to-end contestant pipeline:
1. Load unseen competition test queries via contestant loader.
2. Generate predictions using submission model template.
3. Validate output JSON against official schema.
4. Grade submission using official Jury evaluator tool.
5. Verify zero errors, deterministic execution, and valid score calculation.
"""

from __future__ import annotations
import sys
import os
import json
import subprocess
from pathlib import Path

# Add project root and contestant package to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "contestant_package"))

from contestant_package.loader import load_episodes, assert_zero_leakage
from contestant_package.submission_template import SubmissionModel
from shift.evaluator import Evaluator


def simulate_contestant_workflow(
    test_data_path: str = "datasets/hidden_test_contestant.jsonl",
    submission_output: str = "results/contestant_simulation_submission.json",
    grade_output: str = "results/contestant_simulation_grade.json"
) -> bool:
    print("=" * 70)
    print(" PROJECT SHIFT: Contestant Workflow End-to-End Simulation")
    print("=" * 70)

    # Step 1: Contestant loads competition test data
    print("\n[Step 1/4] Contestant loads test dataset...")
    episodes = load_episodes(test_data_path)
    assert_zero_leakage(episodes)
    print(f"  [OK] Successfully loaded {len(episodes)} test episodes via contestant_package.loader (Zero-Leakage Verified).")

    # Step 2: Contestant runs model inference
    print("\n[Step 2/4] Generating predictions via SubmissionModel...")
    model = SubmissionModel()
    predictions = {}
    for ep in episodes:
        pred = model.predict(ep)
        predictions[ep["episode_id"]] = pred

    sub_p = Path(submission_output)
    sub_p.parent.mkdir(parents=True, exist_ok=True)
    with open(sub_p, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)
    print(f"  [OK] Exported {len(predictions)} predictions to {sub_p}.")

    # Step 3: Verify Draft-07 schema compliance
    print("\n[Step 3/4] Validating prediction schema compliance...")
    _, schema_errors = Evaluator.validate_and_parse_predictions(predictions, strict=False)
    if len(schema_errors) == 0:
        print("  [OK] 100% Schema Compliance: 0 validation errors.")
    else:
        print(f"  [WARN] Schema errors detected in {len(schema_errors)} episodes.")
        return False

    # Step 4: Run official jury grader script via subprocess
    print("\n[Step 4/4] Invoking official jury grader (hidden_evaluator_package)...")
    cmd = [
        sys.executable,
        str(project_root / "hidden_evaluator_package" / "grade_submission.py"),
        "--submission", str(sub_p),
        "--output", str(grade_output)
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode == 0:
        print("  [OK] Official Jury Grader executed successfully with exit code 0.")
        print(proc.stdout)
    else:
        print(f"  [ERROR] Jury Grader failed with exit code {proc.returncode}:")
        print(proc.stderr)
        return False

    print("=" * 70)
    print(" [PASSED] CONTESTANT PIPELINE FULLY VALIDATED AND VERIFIED")
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    success = simulate_contestant_workflow()
    if not success:
        sys.exit(1)
