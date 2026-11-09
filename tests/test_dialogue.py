"""
Unit tests for shift.dialogue module.
"""

import unittest
from shift.lexicon import Lexicon
from shift.drift import DriftEngine
from shift.grammar import GrammarEngine
from shift.dialogue import DialogueGenerator, DiscourseState
from shift.ontology import EventFrame, AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity


class TestDialogue(unittest.TestCase):

    def setUp(self):
        self.lex = Lexicon()
        self.drift = DriftEngine()
        self.drifted_lex = self.drift.transform_lexicon(self.lex)
        self.grammar = GrammarEngine(
            self.drifted_lex,
            self.drift.community_profile,
            self.drift.historical_profile
        )
        self.dialogue_gen = DialogueGenerator(self.grammar, seed=42)

    def test_discourse_state_tracking(self):
        state = DiscourseState()
        frame = EventFrame(
            agent=AgentEntity.SCOUT_ALPHA.value, # masculine
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.CRYSTAL_CORE.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value
        )
        state.update(frame)
        self.assertEqual(state.last_agent, AgentEntity.SCOUT_ALPHA.value)
        self.assertEqual(state.masculine_agent, AgentEntity.SCOUT_ALPHA.value)
        self.assertEqual(state.resolve_pronoun("masculine"), AgentEntity.SCOUT_ALPHA.value)
        self.assertEqual(state.resolve_pronoun("neuter"), ObjectEntity.CRYSTAL_CORE.value)

    def test_coreference_dialogue_generation(self):
        context_turns, query_turn = self.dialogue_gen.generate_dialogue_episode(phenomenon="coreference")
        self.assertGreaterEqual(len(context_turns), 1)
        self.assertEqual(query_turn.reference_type_used, "coreference")
        self.assertIn("agent", query_turn.event_frame.to_dict())
        self.assertGreater(len(query_turn.event_frame.resolved_references), 0)

    def test_ellipsis_dialogue_generation(self):
        context_turns, query_turn = self.dialogue_gen.generate_dialogue_episode(phenomenon="ellipsis")
        self.assertGreaterEqual(len(context_turns), 1)
        self.assertEqual(query_turn.reference_type_used, "ellipsis")
        self.assertIn("[elliptic_verb]", query_turn.event_frame.resolved_references)

    def test_correction_dialogue_generation(self):
        context_turns, query_turn = self.dialogue_gen.generate_dialogue_episode(phenomenon="correction")
        self.assertEqual(len(context_turns), 2) # Turn 1 + Turn 2 (Correction)
        self.assertEqual(query_turn.turn_id, 3)


if __name__ == "__main__":
    unittest.main()
