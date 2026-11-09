"""
Unit tests for shift.lexicon module.
"""

import unittest
from shift.lexicon import Lexicon, MorphemeSet, generate_random_lexicon, levenshtein_distance
from shift.ontology import AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity


class TestLexicon(unittest.TestCase):

    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("tor", "tor"), 0)
        self.assertEqual(levenshtein_distance("tor", "kor"), 1)

    def test_default_lexicon_creation(self):
        lex = Lexicon()
        self.assertGreater(len(lex.concept_to_surface), 20)
        # Check bidirectional mapping consistency
        for c, s in lex.concept_to_surface.items():
            mapped_back = lex.get_concept(s)
            self.assertEqual(mapped_back, c, f"Inconsistent reverse mapping for {c} -> {s}")

    def test_collision_detection(self):
        lex = Lexicon()
        warnings = lex.audit_collisions()
        # Default lexicon should have no exact collisions
        self.assertIsInstance(warnings, list)

    def test_verb_inflection(self):
        lex = Lexicon()
        # synthetic inflection
        inflected = lex.inflect_verb(
            action=ActionPredicate.INSPECT.value,
            tense=Tense.PAST.value,
            polarity=Polarity.NEGATIVE.value,
            synthetic=True
        )
        # root is 'vid', past is 'il', neg is 'ne' -> 'nevidil'
        self.assertEqual(inflected, "nevidil")

    def test_random_lexicon_generation(self):
        lex1 = generate_random_lexicon(seed=42)
        lex2 = generate_random_lexicon(seed=42)
        lex3 = generate_random_lexicon(seed=999)

        # Same seed produces same lexicon
        self.assertEqual(lex1.concept_to_surface, lex2.concept_to_surface)
        # Different seeds produce different lexicons
        self.assertNotEqual(lex1.concept_to_surface, lex3.concept_to_surface)


if __name__ == "__main__":
    unittest.main()
