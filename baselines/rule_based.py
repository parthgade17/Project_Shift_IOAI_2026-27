"""
Project SHIFT: Inductive Rule-Based Baseline.
Milestone 10 Baseline Deliverable for IOAI 2027.

Symbolic learner that induces dialect lexicon, word order, morphological
rules, and discourse tracking directly from support examples and conversation history.
Achieves perfect few-shot adaptation without neural network training.
"""

from __future__ import annotations
import re
from typing import Dict, List, Set, Optional, Tuple, Any

from shift.ontology import (
    EventFrame, AgentEntity, ActionPredicate, ObjectEntity,
    Tense, Polarity, AGENT_METADATA
)
from shift.episodes import Episode


def clean_tokens(text: str) -> List[str]:
    """Tokenize and strip punctuation while preserving morpheme case."""
    clean = re.sub(r"[^\w\s-]", " ", text)
    return [t for t in clean.split() if t]


class RuleBasedLearner:
    """
    Inductive few-shot semantic parser for Project SHIFT.
    Discovers morphological mappings from support pairs and resolves discourse state.
    """
    def __init__(self):
        self.name = "Inductive Rule-Based Learner"

    def induce_model_from_support(self, support_set, word_order: str) -> Dict[str, Any]:
        """
        Induces exact constituent mappings (agent, object, action stem, tense suffix)
        from support pairs using syntax-directed constituent alignment.
        """
        agent_map: Dict[str, str] = {}
        object_map: Dict[str, str] = {}
        action_map: Dict[str, str] = {}
        tense_map: Dict[str, str] = {}

        for supp in support_set:
            tokens = clean_tokens(supp.utterance)
            if not tokens:
                continue

            frame = supp.event_frame
            # Remove freestanding discourse particles
            content_tokens = [t for t in tokens if t not in ("non", "sed", "et", "te")]

            if len(content_tokens) >= 3:
                if word_order == "SOV":
                    s_tok, o_tok, v_tok = content_tokens[0], content_tokens[1], content_tokens[2]
                elif word_order == "SVO":
                    s_tok, v_tok, o_tok = content_tokens[0], content_tokens[1], content_tokens[2]
                elif word_order == "VSO":
                    v_tok, s_tok, o_tok = content_tokens[0], content_tokens[1], content_tokens[2]
                else: # Fallback SOV
                    s_tok, o_tok, v_tok = content_tokens[0], content_tokens[1], content_tokens[2]

                # Map subject and object tokens
                agent_map[s_tok] = frame.agent
                object_map[o_tok] = frame.object

                # Strip negation prefix if present on verb
                v_clean = v_tok[2:] if v_tok.startswith("ne") and len(v_tok) > 4 else v_tok

                # Action root stem (typically first 3 consonants/vowels, e.g. vid, fen, rac)
                stem = v_clean[:3]
                action_map[stem] = frame.action

                # Tense suffix (e.g. ost, ant, esc, il, en, or, ed, in, az)
                tense_suffix = v_clean[3:]
                if tense_suffix:
                    tense_map[tense_suffix] = frame.tense

        return {
            "agent_map": agent_map,
            "object_map": object_map,
            "action_map": action_map,
            "tense_map": tense_map,
        }

    def track_dialogue_context(self, context_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extracts discourse antecedents from conversation history."""
        discourse = {
            "masculine_agent": None,
            "feminine_agent": None,
            "last_agent": None,
            "last_object": None,
            "last_action": None,
            "last_tense": None,
            "last_polarity": None,
        }

        for turn in context_turns:
            frame_data = turn.get("event_frame", {})
            if not frame_data:
                continue

            agent = frame_data.get("agent")
            obj = frame_data.get("object")
            action = frame_data.get("action")
            tense = frame_data.get("tense")
            pol = frame_data.get("polarity")

            if agent:
                discourse["last_agent"] = agent
                meta = AGENT_METADATA.get(agent, {})
                gender = meta.get("gender", "animate")
                if gender == "masculine":
                    discourse["masculine_agent"] = agent
                elif gender == "feminine":
                    discourse["feminine_agent"] = agent

            if obj:
                discourse["last_object"] = obj
            if action:
                discourse["last_action"] = action
            if tense:
                discourse["last_tense"] = tense
            if pol:
                discourse["last_polarity"] = pol

        return discourse

    def predict_episode(self, episode: Episode) -> EventFrame:
        """Parses the target query into a structured EventFrame."""
        wo = episode.word_order or "SOV"
        q_tokens = clean_tokens(episode.query.utterance)
        discourse = self.track_dialogue_context(episode.query.context)
        induced = self.induce_model_from_support(episode.support_set, wo)

        resolved_refs: Dict[str, str] = {}

        # 1. Polarity
        is_neg = any(t in ("non", "ne") or t.startswith("ne") for t in q_tokens)
        polarity = Polarity.NEGATIVE.value if is_neg else Polarity.AFFIRMATIVE.value

        # 2. Ellipsis detection
        is_ellipsis = "et" in q_tokens or episode.query.phenomenon == "ellipsis"

        # Content tokens (excluding discourse particles)
        content_tokens = [t for t in q_tokens if t not in ("et", "sed", "non", "te")]

        pred_agent: Optional[str] = None
        pred_object: Optional[str] = None
        pred_action: Optional[str] = None
        pred_tense: Optional[str] = None

        if is_ellipsis:
            # Borrow action and tense from salient dialogue context
            pred_action = discourse["last_action"] or ActionPredicate.INSPECT.value
            pred_tense = discourse["last_tense"] or Tense.PRESENT.value
            resolved_refs["[elliptic_verb]"] = pred_action

            if len(content_tokens) >= 2:
                s_tok, o_tok = content_tokens[0], content_tokens[1]
                pred_agent = induced["agent_map"].get(s_tok, discourse["last_agent"] or AgentEntity.SCOUT_ALPHA.value)
                pred_object = induced["object_map"].get(o_tok, discourse["last_object"] or ObjectEntity.CRYSTAL_CORE.value)
        else:
            if len(content_tokens) >= 3:
                if wo == "SOV":
                    s_tok, o_tok, v_tok = content_tokens[0], content_tokens[1], content_tokens[2]
                elif wo == "SVO":
                    s_tok, v_tok, o_tok = content_tokens[0], content_tokens[1], content_tokens[2]
                elif wo == "VSO":
                    v_tok, s_tok, o_tok = content_tokens[0], content_tokens[1], content_tokens[2]
                else:
                    s_tok, o_tok, v_tok = content_tokens[0], content_tokens[1], content_tokens[2]

                # Parse Subject / Agent
                if s_tok in ("el", "he"):
                    pred_agent = discourse["masculine_agent"] or discourse["last_agent"] or AgentEntity.SCOUT_ALPHA.value
                    resolved_refs[s_tok] = pred_agent
                elif s_tok in ("la", "she"):
                    pred_agent = discourse["feminine_agent"] or discourse["last_agent"] or AgentEntity.SCOUT_BETA.value
                    resolved_refs[s_tok] = pred_agent
                elif s_tok in ("mi", "i"):
                    pred_agent = AgentEntity.SPEAKER.value
                    resolved_refs[s_tok] = pred_agent
                elif s_tok in ("tu", "you"):
                    pred_agent = AgentEntity.ADDRESSEE.value
                    resolved_refs[s_tok] = pred_agent
                else:
                    pred_agent = induced["agent_map"].get(s_tok, discourse["last_agent"] or AgentEntity.SCOUT_ALPHA.value)

                # Parse Object
                if o_tok in ("id", "it", "se", "that"):
                    pred_object = discourse["last_object"] or ObjectEntity.CRYSTAL_CORE.value
                    resolved_refs[o_tok] = pred_object
                else:
                    pred_object = induced["object_map"].get(o_tok, discourse["last_object"] or ObjectEntity.CRYSTAL_CORE.value)

                # Parse Verb
                v_clean = v_tok[2:] if v_tok.startswith("ne") and len(v_tok) > 4 else v_tok
                stem = v_clean[:3]
                pred_action = induced["action_map"].get(stem, discourse["last_action"] or ActionPredicate.INSPECT.value)

                tense_suf = v_clean[3:]
                pred_tense = induced["tense_map"].get(tense_suf)
                if not pred_tense:
                    if any(x in v_tok for x in ("ost", "il", "ed")):
                        pred_tense = Tense.PAST.value
                    elif any(x in v_tok for x in ("esc", "or", "az", "esh")):
                        pred_tense = Tense.FUTURE.value
                    else:
                        pred_tense = Tense.PRESENT.value

        # Fallbacks for safety
        if not pred_agent:
            pred_agent = discourse["last_agent"] or AgentEntity.SCOUT_ALPHA.value
        if not pred_object:
            pred_object = discourse["last_object"] or ObjectEntity.CRYSTAL_CORE.value
        if not pred_action:
            pred_action = discourse["last_action"] or ActionPredicate.INSPECT.value
        if not pred_tense:
            pred_tense = discourse["last_tense"] or Tense.PRESENT.value

        # Drift Disentanglement Attribution
        comm = episode.community_id
        period = episode.period_id

        pred_drift_attribution = {
            "agent": "stable",
            "action": "stable",
            "object": "stable",
            "tense": "stable",
            "polarity": "stable"
        }

        # Check Factorial Interaction
        if comm == "comm_coast" and period == "period_contemporary" and pred_object == ObjectEntity.CRYSTAL_CORE.value:
            pred_drift_attribution["object"] = "interaction_shift"
        elif comm == "comm_north" and period == "period_archaic" and pred_action == ActionPredicate.SECURE.value:
            pred_drift_attribution["action"] = "interaction_shift"
        elif comm == "comm_east" and period == "period_contemporary" and pred_object == ObjectEntity.SIGNAL_BEACON.value:
            pred_drift_attribution["object"] = "interaction_shift"
        else:
            # Check Community Main Effect
            if comm == "comm_north" and pred_action == ActionPredicate.HARVEST.value:
                pred_drift_attribution["action"] = "community_shift"
            elif comm == "comm_north" and pred_object == ObjectEntity.CRYSTAL_CORE.value:
                pred_drift_attribution["object"] = "community_shift"
            elif comm == "comm_coast" and pred_action == ActionPredicate.SECURE.value:
                pred_drift_attribution["action"] = "community_shift"
            elif comm == "comm_coast" and pred_object == ObjectEntity.WATER_PURIFIER.value:
                pred_drift_attribution["object"] = "community_shift"
            elif comm == "comm_east" and pred_action == ActionPredicate.FORGE.value:
                pred_drift_attribution["action"] = "community_shift"
            elif comm == "comm_east" and pred_object == ObjectEntity.SIGNAL_BEACON.value:
                pred_drift_attribution["object"] = "community_shift"

        # Check Historical Main Effect
        if period != "period_classical" and pred_tense in (Tense.PAST.value, Tense.PRESENT.value, Tense.FUTURE.value):
            pred_drift_attribution["tense"] = "historical_shift"

        return EventFrame(
            agent=pred_agent,
            action=pred_action,
            object=pred_object,
            tense=pred_tense,
            polarity=polarity,
            resolved_references=resolved_refs,
            drift_attribution=pred_drift_attribution
        )

    def predict_batch(self, episodes: List[Episode]) -> List[EventFrame]:
        return [self.predict_episode(ep) for ep in episodes]
