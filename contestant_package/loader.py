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


def stream_episodes(file_path: str) -> Iterator[Dict[str, Any]]:
    """Iterates through episodes one-by-one to conserve memory."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "sample_data/sample_dev.jsonl"
    eps = load_episodes(path)
    print(f"Successfully loaded {len(eps)} episodes from {path}")
    if eps:
        print("Sample Episode Metadata:", eps[0]["episode_id"], eps[0]["community_id"], eps[0]["period_id"])
        print("Query Utterance        :", eps[0]["query"]["utterance"])
