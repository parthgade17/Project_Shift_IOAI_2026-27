"""
Unit tests for shift.grammar module.
"""

import unittest
from shift.lexicon import Lexicon
from shift.drift import DriftEngine, CommunityId, HistoricalPeriod
from shift.grammar import GrammarEngine
from shift.ontology import EventFrame, AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity


class TestGrammar(unittest.TestCase):

    def test_sentence_generation_sov(self):
        lex = Lexicon()
        drift = DriftEngine(community_id=CommunityId.CITADEL.value, period_id=HistoricalPeriod.CLASSICAL.value)
        drifted_lex = drift.transform_lexicon(lex)
        grammar = GrammarEngine(drifted_lex, drift.community_profile, drift.historical_profile)

        frame = EventFrame(
            agent=AgentEntity.SCOUT_ALPHA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.CRYSTAL_CORE.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value
        )
        utt, refs = grammar.generate_sentence(frame)
        # Citadel is SOV: Subject Object Verb. -> "toral kris vidil."
        self.assertEqual(utt, "toral kris vidil.")

    def test_sentence_generation_svo(self):
        lex = Lexicon()
        drift = DriftEngine(community_id=CommunityId.COASTAL_VALLEY.value, period_id=HistoricalPeriod.CLASSICAL.value)
        drifted_lex = drift.transform_lexicon(lex)
        grammar = GrammarEngine(drifted_lex, drift.community_profile, drift.historical_profile)

        frame = EventFrame(
            agent=AgentEntity.SCOUT_ALPHA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.CRYSTAL_CORE.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value
        )
        utt, refs = grammar.generate_sentence(frame)
        # Coastal is SVO with particle negation style
        words = utt.rstrip(".").split()
        self.assertEqual(len(words), 3) # Subj, Verb, Obj

    def test_verb_parsing(self):
        lex = Lexicon()
        drift = DriftEngine(community_id=CommunityId.CITADEL.value, period_id=HistoricalPeriod.CLASSICAL.value)
        drifted_lex = drift.transform_lexicon(lex)
        grammar = GrammarEngine(drifted_lex, drift.community_profile, drift.historical_profile)

        # "nevidil" -> action: inspect, tense: past, polarity: negative
        parsed = grammar.parse_inflected_verb("nevidil")
        self.assertIsNotNone(parsed)
        action, tense, polarity = parsed
        self.assertEqual(action, ActionPredicate.INSPECT.value)
        self.assertEqual(tense, Tense.PAST.value)
        self.assertEqual(polarity, Polarity.NEGATIVE.value)


if __name__ == "__main__":
    unittest.main()
