"""
Project SHIFT: Contestant Submission Template.
IOAI 2027 Competition Starter Kit.

INSTRUCTIONS:
1. Implement your solution inside the `predict(self, episode)` method.
2. Ensure your output dictionary matches the exact EventFrame schema:
   {
       "agent": str,               # e.g. "scout_alpha", "artisan_gamma"
       "action": str,              # e.g. "inspect", "harvest", "forge"
       "object": str,              # e.g. "crystal_core", "signal_beacon"
       "tense": str,               # "past", "present", or "future"
       "polarity": str,            # "affirmative" or "negative"
       "resolved_references": dict # e.g. {"el": "scout_alpha", "id": "crystal_core"}
   }
3. Run `python local_evaluator.py` to test your model locally.
"""

import re
from typing import Dict, Any, List


class SubmissionModel:
    """Contestant model for Project SHIFT."""

    def __init__(self):
        # Initialize internal structures, embeddings, or grammars here
        pass

    def predict(self, episode: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict the structured event frame for the given episode.

        Parameters:
            episode (dict): Contains:
                - "episode_id": str
                - "community_id": str ("comm_north", "comm_coast", "comm_east", "comm_citadel")
                - "period_id": str ("period_archaic", "period_classical", "period_contemporary")
                - "word_order": str ("SOV", "SVO", "VSO")
                - "support_set": list of labelled support examples
                - "query": dict with "utterance" and "context" (dialogue turns)

        Returns:
            dict: Structured EventFrame
        """
        query_utt = episode["query"]["utterance"]
        support_set = episode.get("support_set", [])
        context_turns = episode["query"].get("context", [])
        word_order = episode.get("word_order", "SOV")

        tokens = re.findall(r"\b\w+\b", query_utt.lower())

        # Baseline Starter Heuristic:
        # Step 1: Align words from support set
        agent_map = {}
        object_map = {}
        action_map = {}
        tense_map = {}

        for supp in support_set:
            s_toks = [t for t in re.findall(r"\b\w+\b", supp["utterance"].lower()) if t not in ("non", "sed", "et", "te")]
            f = supp["event_frame"]
            if len(s_toks) >= 3:
                if word_order == "SOV":
                    s_tok, o_tok, v_tok = s_toks[0], s_toks[1], s_toks[2]
                elif word_order == "SVO":
                    s_tok, v_tok, o_tok = s_toks[0], s_toks[1], s_toks[2]
                elif word_order == "VSO":
                    v_tok, s_tok, o_tok = s_toks[0], s_toks[1], s_toks[2]
                else:
                    s_tok, o_tok, v_tok = s_toks[0], s_toks[1], s_toks[2]

                agent_map[s_tok] = f["agent"]
                object_map[o_tok] = f["object"]

                v_clean = v_tok[2:] if v_tok.startswith("ne") and len(v_tok) > 4 else v_tok
                stem = v_clean[:3]
                action_map[stem] = f["action"]
                tense_suffix = v_clean[3:]
                if tense_suffix:
                    tense_map[tense_suffix] = f["tense"]

        # Step 2: Track salient antecedents in dialogue context from text utterance
        last_agent = None
        last_object = None
        last_action = None
        last_tense = None

        for turn in context_turns:
            tf = turn.get("event_frame", {})
            if tf.get("agent"): last_agent = tf["agent"]
            if tf.get("object"): last_object = tf["object"]
            if tf.get("action"): last_action = tf["action"]
            if tf.get("tense"): last_tense = tf["tense"]
            elif "utterance" in turn:
                turn_toks = re.findall(r"\b\w+\b", turn["utterance"].lower())
                for t in turn_toks:
                    if t in agent_map: last_agent = agent_map[t]
                    if t in object_map: last_object = object_map[t]
                    stem = t[:3]
                    if stem in action_map: last_action = action_map[stem]
                    for suf, t_val in tense_map.items():
                        if t.endswith(suf): last_tense = t_val

        # Step 3: Parse Query
        is_neg = any(t in ("non", "ne") or t.startswith("ne") for t in tokens)
        polarity = "negative" if is_neg else "affirmative"

        resolved_refs = {}
        content_tokens = [t for t in tokens if t not in ("non", "sed", "et", "te")]

        # Ellipsis
        if "et" in tokens or len(content_tokens) == 2:
            pred_action = last_action or "inspect"
            pred_tense = last_tense or "present"
            resolved_refs["[elliptic_verb]"] = pred_action
            pred_agent = agent_map.get(content_tokens[0], last_agent or "scout_alpha")
            pred_object = object_map.get(content_tokens[1], last_object or "crystal_core")
        elif len(content_tokens) >= 3:
            if word_order == "SOV":
                s_tok, o_tok, v_tok = content_tokens[0], content_tokens[1], content_tokens[2]
            elif word_order == "SVO":
                s_tok, v_tok, o_tok = content_tokens[0], content_tokens[1], content_tokens[2]
            elif word_order == "VSO":
                v_tok, s_tok, o_tok = content_tokens[0], content_tokens[1], content_tokens[2]
            else:
                s_tok, o_tok, v_tok = content_tokens[0], content_tokens[1], content_tokens[2]

            # Agent / Pronoun
            if s_tok in ("el", "la", "mi", "tu", "he", "she", "i", "you"):
                pred_agent = last_agent or "scout_alpha"
                resolved_refs[s_tok] = pred_agent
            else:
                pred_agent = agent_map.get(s_tok, last_agent or "scout_alpha")

            # Object / Pronoun
            if o_tok in ("id", "it", "se", "that"):
                pred_object = last_object or "crystal_core"
                resolved_refs[o_tok] = pred_object
            else:
                pred_object = object_map.get(o_tok, last_object or "crystal_core")

            # Verb
            v_clean = v_tok[2:] if v_tok.startswith("ne") and len(v_tok) > 4 else v_tok
            stem = v_clean[:3]
            pred_action = action_map.get(stem, last_action or "inspect")
            suf = v_clean[3:]
            pred_tense = tense_map.get(suf)
            if not pred_tense:
                if any(x in v_tok for x in ("ost", "il", "ed")): pred_tense = "past"
                elif any(x in v_tok for x in ("esc", "or", "az", "esh")): pred_tense = "future"
                else: pred_tense = "present"
        else:
            pred_agent = last_agent or "scout_alpha"
            pred_action = last_action or "inspect"
            pred_object = last_object or "crystal_core"
            pred_tense = last_tense or "present"

        # Disentanglement Attribution Heuristic
        comm = episode.get("community_id", "comm_citadel")
        period = episode.get("period_id", "period_classical")
        drift_attribution = {
            "agent": "stable",
            "action": "stable",
            "object": "stable",
            "tense": "historical_shift" if period != "period_classical" else "stable",
            "polarity": "stable"
        }
        if comm == "comm_north" and pred_action == "harvest":
            drift_attribution["action"] = "community_shift"
        elif comm == "comm_coast" and period == "period_contemporary" and pred_object == "crystal_core":
            drift_attribution["object"] = "interaction_shift"

        return {
            "agent": pred_agent,
            "action": pred_action,
            "object": pred_object,
            "tense": pred_tense,
            "polarity": polarity,
            "resolved_references": resolved_refs,
            "drift_attribution": drift_attribution
        }
