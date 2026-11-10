"""
Project SHIFT: JSONL Serialization and Streaming Loader.
Milestone 07 Deliverable for IOAI 2027 Task Development.

Provides robust JSONL dataset export, streaming loading, and secure split
separation between contestant packages and ground-truth answer keys.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Any

from shift.ontology import EventFrame
from shift.episodes import Episode


def export_episodes_jsonl(
    episodes: List[Episode],
    file_path: str | Path,
    strip_labels: bool = False,
    contestant_view: bool = False
) -> None:
    """
    Exports list of episodes to a JSONL file.
    If strip_labels or contestant_view is True, strictly serializes using to_contestant_dict()
    with ZERO query labels, ZERO drift attribution, and ZERO context event_frame annotations.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_contestant = contestant_view or strip_labels
    with open(path, "w", encoding="utf-8") as f:
        for ep in episodes:
            if is_contestant:
                data = ep.to_contestant_dict()
            else:
                data = ep.to_reference_dict()
            f.write(json.dumps(data, ensure_ascii=False) + "\n")


def load_episodes_jsonl(file_path: str | Path) -> List[Episode]:
    """Loads all episodes from a JSONL file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    episodes: List[Episode] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                ep = Episode.from_dict(data)
                episodes.append(ep)
            except Exception as e:
                raise ValueError(f"Error parsing JSONL at {file_path}:{line_num}: {e}")
    return episodes


def stream_episodes_jsonl(file_path: str | Path) -> Iterator[Episode]:
    """Memory-efficient streaming generator for large episode files."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Episode.from_dict(json.loads(line))


def export_ground_truth_labels(
    episodes: List[Episode],
    file_path: str | Path
) -> None:
    """Exports secret answer key mapping episode_id -> EventFrame."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels: Dict[str, Any] = {}
    for ep in episodes:
        if ep.query_label is not None:
            labels[ep.episode_id] = ep.query_label.to_dict()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


def load_ground_truth_labels(file_path: str | Path) -> Dict[str, EventFrame]:
    """Loads ground-truth answer key."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return {ep_id: EventFrame.from_dict(frame_data) for ep_id, frame_data in raw.items()}
