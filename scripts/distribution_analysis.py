"""
Project SHIFT: Dataset Distribution & Balance Analysis.
Component of v0.4 Dataset & Annotation Validation.

Examines label balance, difficulty level representation, dialogue phenomenon
diversity, and community/epoch coverage to verify absence of demographic skews.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shift.serialize import load_episodes_jsonl


def run_distribution_analysis(
    dataset_paths: List[str] = [
        "datasets/train.jsonl",
        "datasets/public_dev.jsonl",
        "datasets/hidden_test_full_reference.jsonl"
    ],
    output_path: str = "results/distribution_analysis_results.json"
) -> Dict[str, Any]:
    print("=" * 70)
    print(" PROJECT SHIFT: Dataset Distribution & Balance Audit")
    print("=" * 70)

    summary: Dict[str, Any] = {}

    for path_str in dataset_paths:
        p = Path(path_str)
        if not p.exists():
            continue

        episodes = load_episodes_jsonl(p)
        print(f"\nAnalyzing: {p.name} (N={len(episodes)} episodes)")

        agent_counts = Counter()
        action_counts = Counter()
        object_counts = Counter()
        tense_counts = Counter()
        polarity_counts = Counter()
        phenom_counts = Counter()
        level_counts = Counter()
        comm_counts = Counter()
        period_counts = Counter()

        for ep in episodes:
            level_counts[f"Level_{ep.difficulty_level}"] += 1
            phenom_counts[ep.query.phenomenon] += 1
            comm_counts[ep.community_id] += 1
            period_counts[ep.period_id] += 1

            if ep.query_label:
                agent_counts[ep.query_label.agent] += 1
                action_counts[ep.query_label.action] += 1
                object_counts[ep.query_label.object] += 1
                tense_counts[ep.query_label.tense] += 1
                polarity_counts[ep.query_label.polarity] += 1

        print("  * Difficulty Levels : " + ", ".join(f"{k}: {v}" for k, v in sorted(level_counts.items())))
        print("  * Phenomena         : " + ", ".join(f"{k}: {v}" for k, v in sorted(phenom_counts.items())))
        print("  * Tense Balance     : " + ", ".join(f"{k}: {v}" for k, v in sorted(tense_counts.items())))
        print("  * Polarity Balance  : " + ", ".join(f"{k}: {v}" for k, v in sorted(polarity_counts.items())))

        summary[p.name] = {
            "total_episodes": len(episodes),
            "difficulty_levels": dict(level_counts),
            "phenomena": dict(phenom_counts),
            "agents": dict(agent_counts),
            "actions": dict(action_counts),
            "objects": dict(object_counts),
            "tenses": dict(tense_counts),
            "polarities": dict(polarity_counts),
            "communities": dict(comm_counts),
            "periods": dict(period_counts),
        }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[OK] Distribution audit exported to {output_path}\n")
    return summary


if __name__ == "__main__":
    run_distribution_analysis()
