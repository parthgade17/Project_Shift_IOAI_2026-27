"""
Unit tests for shift.ontology module.
"""

import unittest
from shift.ontology import (
    EventFrame, AgentEntity, ActionPredicate, ObjectEntity,
    Tense, Polarity, get_json_schema
)


class TestOntology(unittest.TestCase):

    def test_valid_event_frame(self):
        frame = EventFrame(
            agent=AgentEntity.SCOUT_ALPHA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.CRYSTAL_CORE.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value,
            resolved_references={"he": "scout_alpha"}
        )
        is_valid, errors = frame.validate()
        self.assertTrue(is_valid, f"Validation failed with errors: {errors}")
        self.assertEqual(len(errors), 0)

    def test_invalid_event_frame(self):
        frame = EventFrame(
            agent="alien_invader",
            action="fly",
            object="unknown_item",
            tense="yesterday",
            polarity="maybe"
        )
        is_valid, errors = frame.validate()
        self.assertFalse(is_valid)
        self.assertGreaterEqual(len(errors), 5)

    def test_serialization_roundtrip(self):
        frame1 = EventFrame(
            agent=AgentEntity.ARTISAN_GAMMA.value,
            action=ActionPredicate.FORGE.value,
            object=ObjectEntity.SIGNAL_BEACON.value,
            tense=Tense.FUTURE.value,
            polarity=Polarity.NEGATIVE.value,
            resolved_references={"it": "signal_beacon"}
        )
        data = frame1.to_dict()
        frame2 = EventFrame.from_dict(data)
        self.assertTrue(frame1.exact_match(frame2))

    def test_json_schema(self):
        schema = get_json_schema()
        self.assertIn("properties", schema)
        self.assertIn("agent", schema["properties"])
        self.assertIn("action", schema["properties"])
        self.assertEqual(schema["type"], "object")


if __name__ == "__main__":
    unittest.main()
