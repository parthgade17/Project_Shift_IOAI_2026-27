"""
Project SHIFT: Surface Lookup / Nearest Neighbour Baseline.
Milestone 10 Baseline Deliverable for IOAI 2027.

Finds the support example with the highest token overlap to the query utterance
and transfers its event frame.
"""

from __future__ import annotations
import re
from typing import Dict, List, Set, Optional

from shift.ontology import EventFrame, AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity
from shift.episodes import Episode


def tokenize(text: str) -> Set[str]:
    """Simple alphanumeric lowercased tokenizer."""
    return set(re.findall(r"\b\w+\b", text.lower()))


class SurfaceLookupBaseline:
    """Nearest-neighbour lookup based on Jaccard token overlap with support set."""
    def __init__(self):
        self.name = "Surface Lookup Baseline"

    def predict_episode(self, episode: Episode) -> EventFrame:
        if not episode.support_set:
            return EventFrame(
                agent=AgentEntity.SCOUT_ALPHA.value,
                action=ActionPredicate.INSPECT.value,
                object=ObjectEntity.CRYSTAL_CORE.value,
                tense=Tense.PRESENT.value,
                polarity=Polarity.AFFIRMATIVE.value,
                resolved_references={}
            )

        query_tokens = tokenize(episode.query.utterance)
        best_overlap = -1.0
        best_frame = episode.support_set[0].event_frame

        for supp in episode.support_set:
            supp_tokens = tokenize(supp.utterance)
            union = query_tokens.union(supp_tokens)
            if not union:
                continue
            jaccard = len(query_tokens.intersection(supp_tokens)) / len(union)
            if jaccard > best_overlap:
                best_overlap = jaccard
                best_frame = supp.event_frame

        # Return a copy of the best frame
        return EventFrame(
            agent=best_frame.agent,
            action=best_frame.action,
            object=best_frame.object,
            tense=best_frame.tense,
            polarity=best_frame.polarity,
            resolved_references=dict(best_frame.resolved_references)
        )

    def predict_batch(self, episodes: List[Episode]) -> List[EventFrame]:
        return [self.predict_episode(ep) for ep in episodes]
