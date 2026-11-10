"""
Project SHIFT: Answer Leakage and Contestant Security Verification Suite.
Milestone IOAI 2027 Scientific Task Verification.

Asserts that:
1. Zero gold event_frame annotations exist in contestant-visible dialogue context.
2. Public episode serialization completely strips query_label, drift_attribution, and context annotations.
3. Contestant data loader strictly enforces zero-leakage security.
"""

import json
import unittest
from pathlib import Path

from shift.ontology import (
    EventFrame, AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity
)
from shift.episodes import Episode, SupportExample, QueryInstance, EpisodeGenerator
from shift.serialize import load_episodes_jsonl, export_episodes_jsonl
from contestant_package.loader import load_episodes, assert_zero_leakage


class TestLeakageSecurity(unittest.TestCase):
    """Rigorous security audits verifying zero ground-truth leakage to contestants."""

    def setUp(self):
        self.generator = EpisodeGenerator(base_seed=999)
        self.test_ep = self.generator.generate_single_episode(
            episode_id="sec_test_001",
            phenomenon="coreference"
        )

    def test_dialogue_turn_serialization_strips_gold(self):
        """Verifies that dialogue turn strips event_frame when include_gold=False."""
        from shift.dialogue import DialogueTurn
        frame = EventFrame(
            agent=AgentEntity.SCOUT_ALPHA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.CRYSTAL_CORE.value,
            tense=Tense.PRESENT.value,
            polarity=Polarity.AFFIRMATIVE.value
        )
        turn = DialogueTurn(turn_id=1, speaker="Speaker A", utterance="mora vidi pan.", event_frame=frame)

        # Author mode includes gold
        ref_dict = turn.to_dict(include_gold=True)
        self.assertIn("event_frame", ref_dict)
        self.assertIn("reference_type_used", ref_dict)

        # Contestant public mode strictly excludes gold
        pub_dict = turn.to_public_dict()
        self.assertNotIn("event_frame", pub_dict)
        self.assertNotIn("reference_type_used", pub_dict)
        self.assertEqual(pub_dict["utterance"], "mora vidi pan.")
        self.assertEqual(pub_dict["speaker"], "Speaker A")

    def test_episode_contestant_dict_isolation(self):
        """Verifies that Episode.to_contestant_dict() strips all answer keys and context frames."""
        pub_data = self.test_ep.to_contestant_dict()

        # 1. No top-level answer labels
        self.assertNotIn("query_label", pub_data)
        self.assertNotIn("drift_attribution", pub_data)

        # 2. No gold event frames inside dialogue context turns
        for turn in pub_data["query"]["context"]:
            self.assertNotIn("event_frame", turn, f"Leaked event_frame in query context turn: {turn}")
            self.assertNotIn("reference_type_used", turn, f"Leaked reference_type_used in turn: {turn}")
            self.assertIn("utterance", turn)
            self.assertIn("speaker", turn)

    def test_reference_dict_preserves_gold_ground_truth(self):
        """Verifies that Episode.to_reference_dict() preserves full ground truth for jury evaluation."""
        ref_data = self.test_ep.to_reference_dict()
        self.assertIn("query_label", ref_data)
        self.assertIn("drift_attribution", ref_data)
        self.assertIsNotNone(ref_data["query_label"])

    def test_assert_zero_leakage_detector(self):
        """Tests that assert_zero_leakage passes on clean data and catches injected leaks."""
        clean_ep = self.test_ep.to_contestant_dict()
        self.assertTrue(assert_zero_leakage([clean_ep]))

        # Injected top-level leak
        leaky_ep1 = dict(clean_ep)
        leaky_ep1["query_label"] = {"agent": "scout_alpha"}
        with self.assertRaises(ValueError):
            assert_zero_leakage([leaky_ep1])

        # Injected context turn leak
        leaky_ep2 = json.loads(json.dumps(clean_ep))
        if leaky_ep2["query"]["context"]:
            leaky_ep2["query"]["context"][0]["event_frame"] = {"agent": "scout_alpha"}
            with self.assertRaises(ValueError):
                assert_zero_leakage([leaky_ep2])

    def test_hidden_test_contestant_file_cleanliness(self):
        """Audits datasets/hidden_test_contestant.jsonl to confirm zero answers or context event frames."""
        test_file = Path("datasets/hidden_test_contestant.jsonl")
        if not test_file.exists():
            self.skipTest("datasets/hidden_test_contestant.jsonl not generated yet.")

        episodes = load_episodes(str(test_file))
        self.assertGreater(len(episodes), 0)

        # Run strict zero-leakage security assertion
        self.assertTrue(assert_zero_leakage(episodes))

        for ep in episodes:
            self.assertNotIn("query_label", ep)
            self.assertNotIn("drift_attribution", ep)
            for turn in ep.get("query", {}).get("context", []):
                self.assertNotIn("event_frame", turn)
                self.assertNotIn("reference_type_used", turn)


if __name__ == "__main__":
    unittest.main()
