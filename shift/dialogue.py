"""
Project SHIFT: Dialogue and Contextual Interpretation Engine.
Milestone 06 Deliverable for IOAI 2027 Task Development.

Generates coherent multi-turn conversational contexts exhibiting coreference,
pronominal anaphora, definite descriptions, ellipsis, and dialogue corrections
under deterministic discourse resolution rules.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

from shift.ontology import (
    EventFrame, AgentEntity, AgentRole, ActionPredicate, ObjectEntity,
    Tense, Polarity, ReferenceType, AGENT_METADATA
)
from shift.lexicon import Lexicon
from shift.drift import CommunityProfile, HistoricalProfile
from shift.grammar import GrammarEngine


@dataclass
class DialogueTurn:
    """A single conversational turn in an episode."""
    turn_id: int
    speaker: str
    utterance: str
    event_frame: EventFrame
    reference_type_used: str = "direct"

    def to_dict(self, include_gold: bool = True) -> Dict[str, Any]:
        """Serializes turn. If include_gold is False, strips event_frame and reference_type_used."""
        data: Dict[str, Any] = {
            "turn_id": self.turn_id,
            "speaker": self.speaker,
            "utterance": self.utterance
        }
        if include_gold:
            data["event_frame"] = self.event_frame.to_dict()
            data["reference_type_used"] = self.reference_type_used
        return data

    def to_public_dict(self) -> Dict[str, Any]:
        """Contestant-visible view with zero gold event frame annotations."""
        return self.to_dict(include_gold=False)


@dataclass
class DiscourseState:
    """Deterministic discourse memory tracking referents across dialogue turns."""
    masculine_agent: Optional[str] = None
    feminine_agent: Optional[str] = None
    last_agent: Optional[str] = None
    last_object: Optional[str] = None
    last_action: Optional[str] = None
    last_tense: Optional[str] = None
    last_polarity: Optional[str] = None

    def update(self, frame: EventFrame) -> None:
        """Update discourse state with a new event frame."""
        self.last_agent = frame.agent
        self.last_object = frame.object
        self.last_action = frame.action
        self.last_tense = frame.tense
        self.last_polarity = frame.polarity

        meta = AGENT_METADATA.get(frame.agent, {})
        gender = meta.get("gender", "animate")
        if gender == "masculine":
            self.masculine_agent = frame.agent
        elif gender == "feminine":
            self.feminine_agent = frame.agent

    def resolve_pronoun(self, pronoun_type: str) -> Optional[str]:
        """Resolves pronoun to the salient discourse entity."""
        if pronoun_type == "masculine":
            return self.masculine_agent or self.last_agent
        elif pronoun_type == "feminine":
            return self.feminine_agent or self.last_agent
        elif pronoun_type in ("neuter", "demonstrative"):
            return self.last_object
        elif pronoun_type == "speaker":
            return AgentEntity.SPEAKER.value
        elif pronoun_type == "addressee":
            return AgentEntity.ADDRESSEE.value
        return self.last_agent


class DialogueGenerator:
    """
    Generates deterministic, context-rich multi-turn dialogues leading to a target query utterance.
    Supports coreference, ellipsis, and corrections.
    """
    def __init__(self, grammar: GrammarEngine, seed: Optional[int] = None):
        self.grammar = grammar
        self.rng = random.Random(seed)

    def generate_dialogue_episode(
        self,
        phenomenon: str = "coreference",
        forced_action: Optional[str] = None,
        forced_object: Optional[str] = None
    ) -> Tuple[List[DialogueTurn], DialogueTurn]:
        """
        Generates context turns and a final target query turn based on phenomenon:
        - "coreference": Query uses pronouns referencing preceding entities
        - "ellipsis": Query omits the verb, borrowing it from context
        - "correction": Turn 2 corrects Turn 1; Query refers to the corrected state
        - "standard": Direct nouns without ellipsis

        Returns:
            (context_turns, query_turn)
        """
        discourse = DiscourseState()
        context_turns: List[DialogueTurn] = []

        all_agents = [
            AgentEntity.SCOUT_ALPHA.value, AgentEntity.SCOUT_BETA.value,
            AgentEntity.ARTISAN_GAMMA.value, AgentEntity.ARTISAN_DELTA.value,
            AgentEntity.SCHOLAR_OMEGA.value, AgentEntity.SCHOLAR_KAPPA.value,
            AgentEntity.GUARDIAN_THETA.value, AgentEntity.GUARDIAN_SIGMA.value,
        ]
        all_actions = [a.value for a in ActionPredicate]
        all_objects = [o.value for o in ObjectEntity]
        all_tenses = [t.value for t in Tense]

        agent1 = self.rng.choice(all_agents)
        action1 = self.rng.choice(all_actions)
        obj1 = forced_object or self.rng.choice(all_objects)
        tense1 = self.rng.choice(all_tenses)

        frame1 = EventFrame(
            agent=agent1, action=action1, object=obj1,
            tense=tense1, polarity=Polarity.AFFIRMATIVE.value
        )
        utt1, refs1 = self.grammar.generate_sentence(
            frame1,
            subject_ref=ReferenceType.DIRECT_NAME,
            object_ref=ReferenceType.DIRECT_NAME
        )
        frame1.resolved_references = refs1
        discourse.update(frame1)

        turn1 = DialogueTurn(
            turn_id=1, speaker="Speaker A",
            utterance=utt1, event_frame=frame1,
            reference_type_used="direct"
        )
        context_turns.append(turn1)

        if phenomenon == "coreference":
            # Turn 2 (Query): Same agent or object referred via pronoun
            action2 = forced_action or self.rng.choice([a for a in all_actions if a != action1])
            tense2 = self.rng.choice(all_tenses)

            frame_query = EventFrame(
                agent=agent1, action=action2, object=obj1,
                tense=tense2, polarity=Polarity.AFFIRMATIVE.value
            )
            utt_query, refs_query = self.grammar.generate_sentence(
                frame_query,
                subject_ref=ReferenceType.PRONOUN,
                object_ref=ReferenceType.PRONOUN
            )
            frame_query.resolved_references = refs_query

            query_turn = DialogueTurn(
                turn_id=2, speaker="Speaker B",
                utterance=utt_query, event_frame=frame_query,
                reference_type_used="coreference"
            )
            return context_turns, query_turn

        elif phenomenon == "ellipsis":
            # Query omits verb, introduces new agent & object
            agent2 = self.rng.choice([a for a in all_agents if a != agent1])
            obj2 = forced_object or self.rng.choice([o for o in all_objects if o != obj1])
            ellipsis_act = forced_action or action1

            frame_query = EventFrame(
                agent=agent2, action=ellipsis_act, object=obj2,
                tense=tense1, polarity=Polarity.AFFIRMATIVE.value
            )
            utt_query, refs_query = self.grammar.generate_sentence(
                frame_query,
                subject_ref=ReferenceType.DIRECT_NAME,
                object_ref=ReferenceType.DIRECT_NAME,
                verb_ref=ReferenceType.ELLIPSIS
            )
            # Add conjunction "and" prefix for natural ellipsis: e.g. "et tor-be hyd-om."
            conj = self.grammar.lexicon.morphemes.conjunction_and
            utt_query = f"{conj} {utt_query}"
            frame_query.resolved_references = refs_query

            query_turn = DialogueTurn(
                turn_id=2, speaker="Speaker B",
                utterance=utt_query, event_frame=frame_query,
                reference_type_used="ellipsis"
            )
            return context_turns, query_turn

        elif phenomenon == "correction":
            # Turn 2: Correction turn in context
            corrected_obj = forced_object or self.rng.choice([o for o in all_objects if o != obj1])
            corrected_act = forced_action or self.rng.choice([a for a in all_actions if a != action1])

            corr_frame = EventFrame(
                agent=agent1, action=corrected_act, object=corrected_obj,
                tense=tense1, polarity=Polarity.AFFIRMATIVE.value
            )
            corr_utt, corr_refs = self.grammar.generate_sentence(
                corr_frame,
                subject_ref=ReferenceType.DIRECT_NAME,
                object_ref=ReferenceType.DIRECT_NAME
            )
            corr_part = self.grammar.lexicon.morphemes.correction_particle
            turn2_corr = DialogueTurn(
                turn_id=2, speaker="Speaker B",
                utterance=f"{corr_part} {corr_utt}",
                event_frame=corr_frame,
                reference_type_used="correction"
            )
            context_turns.append(turn2_corr)
            discourse.update(corr_frame)

            # Turn 3 (Query): Confirms corrected action
            tense3 = self.rng.choice(all_tenses)
            frame_query = EventFrame(
                agent=agent1, action=corrected_act, object=corrected_obj,
                tense=tense3, polarity=Polarity.AFFIRMATIVE.value
            )
            utt_query, refs_query = self.grammar.generate_sentence(
                frame_query,
                subject_ref=ReferenceType.PRONOUN,
                object_ref=ReferenceType.PRONOUN
            )
            frame_query.resolved_references = refs_query

            query_turn = DialogueTurn(
                turn_id=3, speaker="Speaker A",
                utterance=utt_query, event_frame=frame_query,
                reference_type_used="anaphoric_after_correction"
            )
            return context_turns, query_turn

        else: # Standard direct reference
            agent2 = self.rng.choice(all_agents)
            action2 = forced_action or self.rng.choice(all_actions)
            obj2 = forced_object or self.rng.choice(all_objects)
            tense2 = self.rng.choice(all_tenses)
            pol2 = self.rng.choice([Polarity.AFFIRMATIVE.value, Polarity.NEGATIVE.value])

            frame_query = EventFrame(
                agent=agent2, action=action2, object=obj2,
                tense=tense2, polarity=pol2
            )
            utt_query, refs_query = self.grammar.generate_sentence(
                frame_query,
                subject_ref=ReferenceType.DIRECT_NAME,
                object_ref=ReferenceType.DIRECT_NAME
            )
            frame_query.resolved_references = refs_query

            query_turn = DialogueTurn(
                turn_id=2, speaker="Speaker B",
                utterance=utt_query, event_frame=frame_query,
                reference_type_used="direct"
            )
            return context_turns, query_turn

    def generate_counterfactual_pair(
        self
    ) -> Tuple[Tuple[List[DialogueTurn], DialogueTurn], Tuple[List[DialogueTurn], DialogueTurn]]:
        """
        Creates a paired counterfactual test instance.
        Produces two distinct contexts (Context A and Context B) where the target query utterance
        is 100% IDENTICAL, but the correct interpretation diverges based strictly on conversational history.
        """
        all_actions = [a.value for a in ActionPredicate]
        all_objects = [o.value for o in ObjectEntity]
        all_tenses = [t.value for t in Tense]

        # Use masculine agents for consistent pronoun 'el'
        agent_a = AgentEntity.SCOUT_ALPHA.value
        agent_b = AgentEntity.GUARDIAN_THETA.value

        obj_a = self.rng.choice(all_objects)
        obj_b = self.rng.choice([o for o in all_objects if o != obj_a])

        act_intro = self.rng.choice(all_actions)
        act_target = self.rng.choice([a for a in all_actions if a != act_intro])
        tense_intro = self.rng.choice(all_tenses)
        tense_target = self.rng.choice(all_tenses)

        # Context A Turn 1
        frame_a1 = EventFrame(agent=agent_a, action=act_intro, object=obj_a, tense=tense_intro, polarity=Polarity.AFFIRMATIVE.value)
        utt_a1, refs_a1 = self.grammar.generate_sentence(frame_a1)
        frame_a1.resolved_references = refs_a1
        turn_a1 = DialogueTurn(turn_id=1, speaker="Speaker A", utterance=utt_a1, event_frame=frame_a1)

        # Context B Turn 1
        frame_b1 = EventFrame(agent=agent_b, action=act_intro, object=obj_b, tense=tense_intro, polarity=Polarity.AFFIRMATIVE.value)
        utt_b1, refs_b1 = self.grammar.generate_sentence(frame_b1)
        frame_b1.resolved_references = refs_b1
        turn_b1 = DialogueTurn(turn_id=1, speaker="Speaker A", utterance=utt_b1, event_frame=frame_b1)

        # Query Turn (Identical surface utterance: pronoun 'el' + pronoun 'id' + inflected verb)
        frame_query_a = EventFrame(agent=agent_a, action=act_target, object=obj_a, tense=tense_target, polarity=Polarity.AFFIRMATIVE.value)
        utt_query, refs_query_a = self.grammar.generate_sentence(
            frame_query_a,
            subject_ref=ReferenceType.PRONOUN,
            object_ref=ReferenceType.PRONOUN
        )
        frame_query_a.resolved_references = refs_query_a

        frame_query_b = EventFrame(agent=agent_b, action=act_target, object=obj_b, tense=tense_target, polarity=Polarity.AFFIRMATIVE.value)
        # Use exact same surface pronouns
        refs_query_b = {}
        for k in refs_query_a:
            if refs_query_a[k] == agent_a:
                refs_query_b[k] = agent_b
            elif refs_query_a[k] == obj_a:
                refs_query_b[k] = obj_b
            else:
                refs_query_b[k] = refs_query_a[k]
        frame_query_b.resolved_references = refs_query_b

        query_turn_a = DialogueTurn(turn_id=2, speaker="Speaker B", utterance=utt_query, event_frame=frame_query_a, reference_type_used="counterfactual_pair")
        query_turn_b = DialogueTurn(turn_id=2, speaker="Speaker B", utterance=utt_query, event_frame=frame_query_b, reference_type_used="counterfactual_pair")

        return ([turn_a1], query_turn_a), ([turn_b1], query_turn_b)
