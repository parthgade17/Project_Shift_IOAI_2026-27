"""
Unit tests for shift.drift module.
"""

import unittest
from shift.lexicon import Lexicon
from shift.drift import DriftEngine, CommunityId, HistoricalPeriod
from shift.ontology import ActionPredicate, ObjectEntity, Tense


class TestDrift(unittest.TestCase):

    def test_northern_highlands_drift(self):
        base_lex = Lexicon()
        drift = DriftEngine(
            community_id=CommunityId.NORTHERN_HIGHLANDS.value,
            period_id=HistoricalPeriod.CLASSICAL.value
        )
        drifted_lex = drift.transform_lexicon(base_lex)

        # In Northern Highlands: 'harvest' substituted with 'kor', 'k' shifts to 'c' -> 'cor'
        harvest_surface = drifted_lex.get_surface(ActionPredicate.HARVEST.value)
        self.assertEqual(harvest_surface, "cor")

    def test_archaic_epoch_drift(self):
        base_lex = Lexicon()
        drift = DriftEngine(
            community_id=CommunityId.CITADEL.value,
            period_id=HistoricalPeriod.ARCHAIC.value
        )
        drifted_lex = drift.transform_lexicon(base_lex)

        # Archaic past tense is 'ost'
        past_affix = drifted_lex.morphemes.tense_affixes[Tense.PAST.value]
        self.assertEqual(past_affix, "ost")

    def test_word_order_preference(self):
        drift_east = DriftEngine(community_id=CommunityId.EASTERN_PLAINS.value)
        self.assertEqual(drift_east.get_word_order(), "VSO")

        drift_coast = DriftEngine(community_id=CommunityId.COASTAL_VALLEY.value)
        self.assertEqual(drift_coast.get_word_order(), "SVO")


if __name__ == "__main__":
    unittest.main()
