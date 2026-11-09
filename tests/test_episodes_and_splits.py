"""
Unit tests for shift.episodes, shift.splits, and shift.serialize modules.
"""

import unittest
import tempfile
import os
from pathlib import Path

from shift.episodes import EpisodeGenerator
from shift.splits import SplitManager
from shift.serialize import (
    export_episodes_jsonl, load_episodes_jsonl,
    export_ground_truth_labels, load_ground_truth_labels
)


class TestEpisodesAndSplits(unittest.TestCase):

    def test_single_episode_generation(self):
        gen = EpisodeGenerator(base_seed=123)
        ep = gen.generate_single_episode(
            episode_id="test_ep_01",
            k_support=8,
            phenomenon="coreference"
        )
        self.assertEqual(ep.episode_id, "test_ep_01")
        self.assertEqual(len(ep.support_set), 8)
        self.assertIsNotNone(ep.query_label)
        self.assertTrue(ep.query_label.validate()[0])

    def test_jsonl_export_and_load(self):
        gen = EpisodeGenerator(base_seed=456)
        batch = gen.generate_batch(count=5, k_support=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test_episodes.jsonl"
            labels_path = Path(tmpdir) / "test_labels.json"

            # Export with labels
            export_episodes_jsonl(batch, file_path, strip_labels=False)
            loaded = load_episodes_jsonl(file_path)
            self.assertEqual(len(loaded), 5)
            self.assertEqual(loaded[0].episode_id, batch[0].episode_id)
            self.assertEqual(loaded[0].query_label.agent, batch[0].query_label.agent)

            # Export secret labels
            export_ground_truth_labels(batch, labels_path)
            labels = load_ground_truth_labels(labels_path)
            self.assertEqual(len(labels), 5)
            self.assertIn("ep_0001", labels)

    def test_splits_and_leakage_audit(self):
        manager = SplitManager(base_seed=777)
        splits = manager.create_partitioned_dataset(
            train_count=10,
            val_count=4,
            dev_count=4,
            test_iid_count=4,
            test_ood_count=4,
            test_comm_count=2,
            test_period_count=2,
            k_support=4
        )
        self.assertEqual(len(splits.train), 10)
        self.assertEqual(len(splits.hidden_test), 12)

        audit = manager.audit_leakage(splits)
        self.assertTrue(audit["leak_free"])
        self.assertEqual(len(audit["leakage_test_in_train"]), 0)


if __name__ == "__main__":
    unittest.main()
