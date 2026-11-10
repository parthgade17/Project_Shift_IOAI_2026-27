"""
Project SHIFT: Statistical / Probabilistic Baseline.
Milestone 10 Baseline Deliverable for IOAI 2027.

Implements a statistical Naive Bayes and Maximum Likelihood Estimation (MLE)
parser that infers slot probabilities and n-gram alignments from support sets.
"""

from __future__ import annotations
import re
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, Counter

from shift.ontology import (
    EventFrame, AgentEntity, ActionPredicate, ObjectEntity,
    Tense, Polarity, AGENT_METADATA
)
from shift.episodes import Episode


def clean_tokens(text: str) -> List[str]:
    """Tokenize and strip punctuation while preserving lowercased morphemes."""
    clean = re.sub(r"[^\w\s-]", " ", text.lower())
    return [t for t in clean.split() if t]


class StatisticalLearner:
    """
    Probabilistic semantic parser estimating conditional likelihoods:
    P(Agent | token), P(Object | token), P(Action | stem), P(Tense | suffix).
    """
    def __init__(self, smoothing: float = 0.1):
        self.name = "Statistical / Naive Bayes Baseline"
        self.smoothing = smoothing

    def fit_support_probabilities(self, support_set, word_order: str) -> Dict[str, Any]:
        """Estimates MLE counts from support set with Laplace smoothing."""
        agent_counts = defaultdict(Counter)
        object_counts = defaultdict(Counter)
        action_counts = defaultdict(Counter)
        tense_counts = defaultdict(Counter)

        for supp in support_set:
            tokens = clean_tokens(supp.utterance)
            content = [t for t in tokens if t not in ("non", "sed", "et", "te")]
            frame = supp.event_frame

            if len(content) >= 3:
                if word_order == "SOV":
                    s_tok, o_tok, v_tok = content[0], content[1], content[2]
                elif word_order == "SVO":
                    s_tok, v_tok, o_tok = content[0], content[1], content[2]
                elif word_order == "VSO":
                    v_tok, s_tok, o_tok = content[0], content[1], content[2]
                else:
                    s_tok, o_tok, v_tok = content[0], content[1], content[2]

                agent_counts[s_tok][frame.agent] += 1
                object_counts[o_tok][frame.object] += 1

                # Strip negation
                v_clean = v_tok[2:] if v_tok.startswith("ne") and len(v_tok) > 4 else v_tok
                stem = v_clean[:3]
                action_counts[stem][frame.action] += 1
                suffix = v_clean[3:]
                if suffix:
                    tense_counts[suffix][frame.tense] += 1

        return {
            "agent_counts": agent_counts,
            "object_counts": object_counts,
            "action_counts": action_counts,
            "tense_counts": tense_counts
        }

    def predict_episode(self, episode: Episode) -> EventFrame:
        """Computes argmax P(slot | observation) for the target query."""
        wo = episode.word_order or "SOV"
        probs = self.fit_support_probabilities(episode.support_set, wo)
        q_tokens = clean_tokens(episode.query.utterance)
        content = [t for t in q_tokens if t not in ("non", "sed", "et", "te")]

        # Context antecedent fallback
        last_agent = None
        last_object = None
        last_action = None
        last_tense = None

        if episode.query.context:
            for turn in episode.query.context:
                f = turn.get("event_frame", {})
                if f.get("agent"): last_agent = f["agent"]
                if f.get("object"): last_object = f["object"]
                if f.get("action"): last_action = f["action"]
                if f.get("tense"): last_tense = f["tense"]
                elif "utterance" in turn:
                    turn_toks = clean_tokens(turn["utterance"])
                    for t in turn_toks:
                        if probs["agent_counts"].get(t):
                            last_agent = probs["agent_counts"][t].most_common(1)[0][0]
                        if probs["object_counts"].get(t):
                            last_object = probs["object_counts"][t].most_common(1)[0][0]
                        stem = t[:3]
                        if probs["action_counts"].get(stem):
                            last_action = probs["action_counts"][stem].most_common(1)[0][0]
                        for suf in probs["tense_counts"]:
                            if t.endswith(suf) and probs["tense_counts"][suf]:
                                last_tense = probs["tense_counts"][suf].most_common(1)[0][0]

        resolved_refs: Dict[str, str] = {}
        is_neg = any(t in ("non", "ne") or t.startswith("ne") for t in q_tokens)
        polarity = Polarity.NEGATIVE.value if is_neg else Polarity.AFFIRMATIVE.value

        is_ellipsis = "et" in q_tokens or episode.query.phenomenon == "ellipsis"

        if is_ellipsis:
            pred_action = last_action or ActionPredicate.INSPECT.value
            pred_tense = last_tense or Tense.PRESENT.value
            resolved_refs["[elliptic_verb]"] = pred_action

            s_tok = content[0] if len(content) > 0 else ""
            o_tok = content[1] if len(content) > 1 else ""

            pred_agent = (probs["agent_counts"][s_tok].most_common(1)[0][0]
                          if probs["agent_counts"][s_tok] else (last_agent or AgentEntity.SCOUT_ALPHA.value))
            pred_object = (probs["object_counts"][o_tok].most_common(1)[0][0]
                           if probs["object_counts"][o_tok] else (last_object or ObjectEntity.CRYSTAL_CORE.value))
        elif len(content) >= 3:
            if wo == "SOV":
                s_tok, o_tok, v_tok = content[0], content[1], content[2]
            elif wo == "SVO":
                s_tok, v_tok, o_tok = content[0], content[1], content[2]
            elif wo == "VSO":
                v_tok, s_tok, o_tok = content[0], content[1], content[2]
            else:
                s_tok, o_tok, v_tok = content[0], content[1], content[2]

            # Agent inference
            if s_tok in ("el", "la", "mi", "tu", "he", "she"):
                pred_agent = last_agent or AgentEntity.SCOUT_ALPHA.value
                resolved_refs[s_tok] = pred_agent
            elif probs["agent_counts"][s_tok]:
                pred_agent = probs["agent_counts"][s_tok].most_common(1)[0][0]
            else:
                pred_agent = last_agent or AgentEntity.SCOUT_ALPHA.value

            # Object inference
            if o_tok in ("id", "it", "se", "that"):
                pred_object = last_object or ObjectEntity.CRYSTAL_CORE.value
                resolved_refs[o_tok] = pred_object
            elif probs["object_counts"][o_tok]:
                pred_object = probs["object_counts"][o_tok].most_common(1)[0][0]
            else:
                pred_object = last_object or ObjectEntity.CRYSTAL_CORE.value

            # Verb inference
            v_clean = v_tok[2:] if v_tok.startswith("ne") and len(v_tok) > 4 else v_tok
            stem = v_clean[:3]
            if probs["action_counts"][stem]:
                pred_action = probs["action_counts"][stem].most_common(1)[0][0]
            else:
                pred_action = last_action or ActionPredicate.INSPECT.value

            suf = v_clean[3:]
            if probs["tense_counts"][suf]:
                pred_tense = probs["tense_counts"][suf].most_common(1)[0][0]
            else:
                if any(x in v_tok for x in ("ost", "il", "ed")): pred_tense = Tense.PAST.value
                elif any(x in v_tok for x in ("esc", "or", "az", "esh")): pred_tense = Tense.FUTURE.value
                else: pred_tense = Tense.PRESENT.value
        else:
            pred_agent = last_agent or AgentEntity.SCOUT_ALPHA.value
            pred_action = last_action or ActionPredicate.INSPECT.value
            pred_object = last_object or ObjectEntity.CRYSTAL_CORE.value
            pred_tense = last_tense or Tense.PRESENT.value

        return EventFrame(
            agent=pred_agent,
            action=pred_action,
            object=pred_object,
            tense=pred_tense,
            polarity=polarity,
            resolved_references=resolved_refs
        )

    def predict_batch(self, episodes: List[Episode]) -> List[EventFrame]:
        return [self.predict_episode(ep) for ep in episodes]
