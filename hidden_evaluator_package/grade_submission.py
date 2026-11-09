"""
IOAI 2027 Scientific Committee & Jury Official Grading Script
Task: Project SHIFT (In-Context Semantic Adaptation & Drift Disentanglement)

Usage:
    python grade_submission.py --submission <contestant_predictions.json>
    python grade_submission.py --submission <contestant_predictions.json> --output grade.json
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to path if available
script_dir = Path(__file__).resolve().parent
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

try:
    from shift.evaluator import Evaluator
    from shift.schema import SchemaValidator
    from shift.serialize import load_ground_truth_labels, load_episodes_jsonl
except ImportError:
    # If running isolated, import local modules or fail with clear instruction
    raise ImportError("Could not import Project SHIFT core library. Ensure shift package is in PYTHONPATH or parent directory.")


def grade_contestant_submission(
    submission_path: str | Path,
    labels_path: str | Path = None,
    dataset_path: str | Path = None,
    output_path: str | Path = None
) -> dict:
    sub_p = Path(submission_path)
    if not sub_p.exists():
        raise FileNotFoundError(f"Contestant submission file not found: {sub_p}")

    if labels_path is None:
        labels_path = script_dir / "hidden_test_labels.json"
    lab_p = Path(labels_path)
    if not lab_p.exists():
        # Fallback to parent datasets directory
        fallback_lab = parent_dir / "datasets" / "hidden_test_labels.json"
        if fallback_lab.exists():
            lab_p = fallback_lab
        else:
            raise FileNotFoundError(f"Hidden test labels key not found: {lab_p}")

    if dataset_path is None:
        cand_ds = parent_dir / "datasets" / "hidden_test_contestant.jsonl"
        if cand_ds.exists():
            dataset_path = cand_ds

    results = Evaluator.evaluate_files(
        predictions_path=sub_p,
        dataset_path=dataset_path,
        labels_path=lab_p,
        output_path=output_path,
        strict_schema=False
    )

    return results


def main():
    parser = argparse.ArgumentParser(description="IOAI 2027 Official Grading Tool - Project SHIFT")
    parser.add_argument("--submission", type=str, required=True, help="Path to contestant submission file (JSON/JSONL)")
    parser.add_argument("--labels", type=str, default=None, help="Path to ground truth labels JSON (defaults to hidden_test_labels.json)")
    parser.add_argument("--dataset", type=str, default=None, help="Path to hidden test dataset JSONL (optional for metadata)")
    parser.add_argument("--output", type=str, default=None, help="Path to save grading result JSON")

    args = parser.parse_args()

    print("\n" + "#" * 68)
    print(" IOAI 2027 OFFICIAL JURY EVALUATION: PROJECT SHIFT")
    print("#" * 68)
    print(f" Contestant Submission : {args.submission}")

    results = grade_contestant_submission(
        submission_path=args.submission,
        labels_path=args.labels,
        dataset_path=args.dataset,
        output_path=args.output
    )

    Evaluator.print_summary(results, title="OFFICIAL IOAI 2027 CONTESTANT GRADE")

    comp_score = results.get("composite_competition_score", 0.0)
    print(f">>> FINAL OFFICIAL SCORE: {comp_score:6.2f} / 100.00 <<<\n")


if __name__ == "__main__":
    main()
