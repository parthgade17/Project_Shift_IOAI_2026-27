"""
Project SHIFT: Majority Class Baseline.
Milestone 10 Baseline Deliverable for IOAI 2027.

Predicts the most frequent slot values observed in the episode's support set.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from collections import Counter

from shift.ontology import EventFrame, AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity
from shift.episodes import Episode


class MajorityBaseline:
    """Predicts majority class from support set."""
    def __init__(self):
        self.name = "Majority Baseline"

    def predict_episode(self, episode: Episode) -> EventFrame:
        if not episode.support_set:
            # Default fallback
            return EventFrame(
                agent=AgentEntity.SCOUT_ALPHA.value,
                action=ActionPredicate.INSPECT.value,
                object=ObjectEntity.CRYSTAL_CORE.value,
                tense=Tense.PRESENT.value,
                polarity=Polarity.AFFIRMATIVE.value,
                resolved_references={}
            )

        agents = Counter(s.event_frame.agent for s in episode.support_set)
        actions = Counter(s.event_frame.action for s in episode.support_set)
        objects = Counter(s.event_frame.object for s in episode.support_set)
        tenses = Counter(s.event_frame.tense for s in episode.support_set)
        polarities = Counter(s.event_frame.polarity for s in episode.support_set)

        return EventFrame(
            agent=agents.most_common(1)[0][0],
            action=actions.most_common(1)[0][0],
            object=objects.most_common(1)[0][0],
            tense=tenses.most_common(1)[0][0],
            polarity=polarities.most_common(1)[0][0],
            resolved_references={}
        )

    def predict_batch(self, episodes: List[Episode]) -> List[EventFrame]:
        return [self.predict_episode(ep) for ep in episodes]
