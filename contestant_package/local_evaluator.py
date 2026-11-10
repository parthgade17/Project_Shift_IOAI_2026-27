"""
Project SHIFT: Local Evaluator for Contestants.
IOAI 2027 Competition.

Usage:
    python local_evaluator.py
    python local_evaluator.py --data path/to/dataset.jsonl
"""

import sys
import json
import argparse
from pathlib import Path

from loader import load_episodes
from submission_template import SubmissionModel


def evaluate_model(model, episodes):
    total = len(episodes)
    if total == 0:
        print("No episodes to evaluate.")
        return

    exact_matches = 0
    disentangle_matches = 0
    slot_matches = {
        "agent": 0, "action": 0, "object": 0,
        "tense": 0, "polarity": 0, "resolved_references": 0
    }
    level_stats = {}

    for ep in episodes:
        pred = model.predict(ep)
        truth = ep.get("query_label")
        if not truth:
            print(f"[WARN] Episode {ep.get('episode_id')} missing query_label for local evaluation.")
            continue

        is_exact = (
            pred.get("agent") == truth.get("agent") and
            pred.get("action") == truth.get("action") and
            pred.get("object") == truth.get("object") and
            pred.get("tense") == truth.get("tense") and
            pred.get("polarity") == truth.get("polarity") and
            pred.get("resolved_references", {}) == truth.get("resolved_references", {})
        )

        if is_exact:
            exact_matches += 1

        for slot in ["agent", "action", "object", "tense", "polarity"]:
            if pred.get(slot) == truth.get(slot):
                slot_matches[slot] += 1
        if pred.get("resolved_references", {}) == truth.get("resolved_references", {}):
            slot_matches["resolved_references"] += 1

        if truth.get("drift_attribution") and pred.get("drift_attribution"):
            if truth["drift_attribution"] == pred["drift_attribution"]:
                disentangle_matches += 1

        lvl = ep.get("difficulty_level", 1)
        if lvl not in level_stats:
            level_stats[lvl] = {"total": 0, "exact": 0}
        level_stats[lvl]["total"] += 1
        if is_exact:
            level_stats[lvl]["exact"] += 1

    print("\n" + "="*50)
    print("LOCAL EVALUATION RESULTS")
    print("="*50)
    print(f"Total Evaluated        : {total}")
    print(f"Exact Frame Accuracy   : {exact_matches/total*100:.2f}%")
    print(f"Disentanglement Acc    : {disentangle_matches/total*100:.2f}%")
    print("-" * 50)
    print("Per-Slot Accuracies:")
    for slot, count in slot_matches.items():
        print(f"  * {slot:<22}: {count/total*100:6.2f}%")

    if level_stats:
        print("-" * 50)
        print("Progressive Difficulty Breakdown:")
        for lvl in sorted(level_stats.keys()):
            st = level_stats[lvl]
            acc = st["exact"] / st["total"] * 100.0 if st["total"] else 0.0
            print(f"  * Level {lvl:<16}: {acc:6.2f}% (N={st['total']})")
    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Project SHIFT submission locally.")
    parser.add_argument("--data", default="sample_data/sample_dev.jsonl", help="Dataset JSONL file")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_path = Path(args.data)
    if not data_path.exists():
        if (script_dir / args.data).exists():
            data_path = script_dir / args.data
        elif (Path("datasets/public_dev.jsonl")).exists():
            data_path = Path("datasets/public_dev.jsonl")
        elif (script_dir.parent / "datasets/public_dev.jsonl").exists():
            data_path = script_dir.parent / "datasets/public_dev.jsonl"
        else:
            print(f"Dataset {args.data} not found. Please provide a valid JSONL file.")
            sys.exit(1)

    print(f"Loading data from {data_path}...")
    episodes = load_episodes(str(data_path))
    model = SubmissionModel()
    evaluate_model(model, episodes)


if __name__ == "__main__":
    main()
