"""
Project SHIFT: Grammar Realisation and Parsing Engine.
Milestone 05 Deliverable for IOAI 2027 Task Development.

Generates grammatical sentences with controlled word order (SOV, SVO, VSO),
morphological marking (tense, polarity, case), and provides reversible syntactic parsing.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

from shift.ontology import (
    EventFrame, AgentEntity, ActionPredicate, ObjectEntity,
    Tense, Polarity, ReferenceType, AGENT_METADATA
)
from shift.lexicon import Lexicon
from shift.drift import CommunityProfile, HistoricalProfile


@dataclass
class SentenceConstituents:
    """Decomposed syntactic elements of a realized sentence."""
    subject_surface: str
    verb_surface: str
    object_surface: str
    tense: str
    polarity: str
    word_order: str
    case_marked: bool = False


class GrammarEngine:
    """
    Realises EventFrames into surface utterances and parses surface utterances
    back into semantic constituent hypotheses.
    """
    def __init__(
        self,
        lexicon: Lexicon,
        community_profile: CommunityProfile,
        historical_profile: HistoricalProfile
    ):
        self.lexicon = lexicon
        self.community_profile = community_profile
        self.historical_profile = historical_profile

    def realise_subject(
        self,
        agent: str,
        ref_type: ReferenceType = ReferenceType.DIRECT_NAME
    ) -> Tuple[str, Optional[Tuple[str, str]]]:
        """
        Realises subject constituent based on reference type.
        Returns (surface_token, Optional[(referring_expression, canonical_agent)]).
        """
        if ref_type == ReferenceType.DIRECT_NAME:
            surface = self.lexicon.get_surface(agent)
            if self.historical_profile.use_case_marking:
                surface = f"{surface}a"
            return surface, None

        elif ref_type == ReferenceType.PRONOUN:
            meta = AGENT_METADATA.get(agent, {"gender": "animate"})
            gender = meta.get("gender", "animate")
            if agent == AgentEntity.SPEAKER.value:
                pron_token = self.lexicon.get_surface("pronoun:speaker")
                return pron_token, (pron_token, agent)
            elif agent == AgentEntity.ADDRESSEE.value:
                pron_token = self.lexicon.get_surface("pronoun:addressee")
                return pron_token, (pron_token, agent)
            elif gender == "masculine":
                pron_token = self.lexicon.get_surface("pronoun:masculine")
                return pron_token, (pron_token, agent)
            elif gender == "feminine":
                pron_token = self.lexicon.get_surface("pronoun:feminine")
                return pron_token, (pron_token, agent)
            else:
                pron_token = self.lexicon.get_surface("pronoun:neuter")
                return pron_token, (pron_token, agent)

        elif ref_type == ReferenceType.DEFINITE_NP:
            meta = AGENT_METADATA.get(agent, {"role": "scout"})
            role = meta.get("role", "scout")
            det = self.lexicon.morphemes.determiner
            role_root = self.lexicon.morphemes.role_roots.get(role, "tor")
            surface = f"{det} {role_root}"
            return surface, (surface, agent)

        elif ref_type == ReferenceType.DEMONSTRATIVE:
            dem = self.lexicon.get_surface("pronoun:demonstrative")
            return dem, (dem, agent)

        elif ref_type == ReferenceType.ELLIPSIS:
            return "", (f"[elliptic_subject]", agent)

        else:
            surface = self.lexicon.get_surface(agent)
            return surface, None

    def realise_object(
        self,
        obj: str,
        ref_type: ReferenceType = ReferenceType.DIRECT_NAME
    ) -> Tuple[str, Optional[Tuple[str, str]]]:
        """Realises object constituent."""
        if ref_type == ReferenceType.DIRECT_NAME:
            surface = self.lexicon.get_surface(obj)
            if self.historical_profile.use_case_marking:
                surface = f"{surface}om"
            return surface, None

        elif ref_type == ReferenceType.PRONOUN or ref_type == ReferenceType.DEMONSTRATIVE:
            pron_token = self.lexicon.get_surface("pronoun:neuter")
            return pron_token, (pron_token, obj)

        elif ref_type == ReferenceType.DEFINITE_NP:
            det = self.lexicon.morphemes.determiner
            root = self.lexicon.get_surface(obj)
            surface = f"{det} {root}"
            return surface, (surface, obj)

        elif ref_type == ReferenceType.ELLIPSIS:
            return "", (f"[elliptic_object]", obj)

        else:
            surface = self.lexicon.get_surface(obj)
            return surface, None

    def realise_verb(self, action: str, tense: str, polarity: str) -> str:
        """Realises inflected verb according to dialect and historical morphology."""
        verb_root = self.lexicon.get_surface(action)
        tense_affix = self.lexicon.morphemes.tense_affixes[tense]
        neg_affix = self.lexicon.morphemes.polarity_affixes.get(Polarity.NEGATIVE.value, "ne")
        neg_part = self.lexicon.morphemes.negation_particle

        # Synthesis
        verb_stem = f"{verb_root}{tense_affix}"

        if polarity == Polarity.NEGATIVE.value:
            if self.community_profile.negation_style == "prefix":
                return f"{neg_affix}{verb_stem}"
            else: # particle
                return f"{verb_stem} {neg_part}"
        return verb_stem

    def generate_sentence(
        self,
        frame: EventFrame,
        subject_ref: ReferenceType = ReferenceType.DIRECT_NAME,
        object_ref: ReferenceType = ReferenceType.DIRECT_NAME,
        verb_ref: ReferenceType = ReferenceType.DIRECT_NAME
    ) -> Tuple[str, Dict[str, str]]:
        """
        Converts EventFrame into surface utterance according to word order rules.
        Returns:
            (utterance_string, resolved_references_dict)
        """
        resolved_refs: Dict[str, str] = dict(frame.resolved_references)

        subj_str, subj_ref_info = self.realise_subject(frame.agent, subject_ref)
        if subj_ref_info:
            ref_surface, target = subj_ref_info
            resolved_refs[ref_surface] = target

        obj_str, obj_ref_info = self.realise_object(frame.object, object_ref)
        if obj_ref_info:
            ref_surface, target = obj_ref_info
            resolved_refs[ref_surface] = target

        if verb_ref == ReferenceType.ELLIPSIS:
            verb_str = ""
            resolved_refs["[elliptic_verb]"] = frame.action
        else:
            verb_str = self.realise_verb(frame.action, frame.tense, frame.polarity)

        word_order = self.community_profile.word_order_preference

        # Assemble sentence constituents based on word order
        tokens: List[str] = []
        if word_order == "SOV":
            if subj_str: tokens.append(subj_str)
            if obj_str: tokens.append(obj_str)
            if verb_str: tokens.append(verb_str)
        elif word_order == "SVO":
            if subj_str: tokens.append(subj_str)
            if verb_str: tokens.append(verb_str)
            if obj_str: tokens.append(obj_str)
        elif word_order == "VSO":
            if verb_str: tokens.append(verb_str)
            if subj_str: tokens.append(subj_str)
            if obj_str: tokens.append(obj_str)
        else: # Default fallback SOV
            if subj_str: tokens.append(subj_str)
            if obj_str: tokens.append(obj_str)
            if verb_str: tokens.append(verb_str)

        sentence = " ".join(tokens).strip() + "."
        return sentence, resolved_refs

    def parse_inflected_verb(self, verb_token: str) -> Optional[Tuple[str, str, str]]:
        """
        Decomposes an inflected verb token into (action, tense, polarity).
        Returns None if parsing fails.
        """
        clean_tok = verb_token.rstrip(".").strip()
        polarity = Polarity.AFFIRMATIVE.value

        # Check for negation particle
        parts = clean_tok.split()
        if len(parts) == 2 and parts[1] == self.lexicon.morphemes.negation_particle:
            polarity = Polarity.NEGATIVE.value
            clean_tok = parts[0]
        elif clean_tok.startswith(self.lexicon.morphemes.polarity_affixes.get(Polarity.NEGATIVE.value, "ne")):
            neg_prefix = self.lexicon.morphemes.polarity_affixes[Polarity.NEGATIVE.value]
            polarity = Polarity.NEGATIVE.value
            clean_tok = clean_tok[len(neg_prefix):]

        # Match tense affix at end
        matched_tense = None
        verb_stem = None
        for tense_val, affix in self.lexicon.morphemes.tense_affixes.items():
            if clean_tok.endswith(affix):
                stem = clean_tok[:-len(affix)]
                # Check if stem matches an action
                concept = self.lexicon.get_concept(stem)
                if concept in [a.value for a in ActionPredicate]:
                    matched_tense = tense_val
                    verb_stem = concept
                    break

        if verb_stem and matched_tense:
            return verb_stem, matched_tense, polarity
        return None
