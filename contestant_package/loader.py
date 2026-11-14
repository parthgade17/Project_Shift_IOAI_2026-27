"""
Project SHIFT: Contestant Data Loader.
IOAI 2027 Starter Kit.

Provides simple utilities to stream and load competition episodes.
Requires only standard library Python 3.9+.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Iterator


def load_episodes(file_path: str) -> List[Dict[str, Any]]:
    """Loads all episodes from a JSONL file into a list of dictionaries."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    episodes = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def assert_zero_leakage(episodes: Any, check_query_labels: bool = True) -> bool:
    """
    Rigorously verifies that contestant-facing episodes contain zero answer leakage:
    1. No top-level query_label or drift_attribution (when check_query_labels is True).
    2. No gold event_frame or reference_type_used inside dialogue context turns.
    """
    if isinstance(episodes, (str, Path)):
        episodes = load_episodes(episodes)
    for ep in episodes:
        if check_query_labels:
            if "query_label" in ep:
                raise ValueError(f"Security Alert: query_label detected in contestant episode {ep.get('episode_id')}")
            if "drift_attribution" in ep:
                raise ValueError(f"Security Alert: drift_attribution detected in contestant episode {ep.get('episode_id')}")

        q_ctx = ep.get("query", {}).get("context", [])
        for t in q_ctx:
            if "event_frame" in t:
                raise ValueError(f"Security Alert: gold event_frame detected in context turn of episode {ep.get('episode_id')}")
            if "reference_type_used" in t:
                raise ValueError(f"Security Alert: reference_type_used detected in context turn of episode {ep.get('episode_id')}")
    return True


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/sample_dev.jsonl"
    eps = load_episodes(path)
    print(f"Successfully loaded {len(eps)} episodes from {path}")
    if eps:
        print("Sample Episode Metadata:", eps[0]["episode_id"], eps[0]["community_id"], eps[0]["period_id"])
        print("Query Utterance        :", eps[0]["query"]["utterance"])
