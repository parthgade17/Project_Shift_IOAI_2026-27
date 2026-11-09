"""
Unit tests for baselines and robustness shortcut tests.
"""

import unittest
from shift.episodes import EpisodeGenerator
from shift.robustness import ShortcutAuditor
from shift.validation import audit_dataset_quality
from baselines.majority import MajorityBaseline
from baselines.surface_lookup import SurfaceLookupBaseline
from baselines.rule_based import RuleBasedLearner
from baselines.evaluate import Evaluator


class TestBaselinesAndRobustness(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        gen = EpisodeGenerator(base_seed=101)
        cls.episodes = gen.generate_batch(count=12, k_support=8)
        cls.ground_truth = {ep.episode_id: ep.query_label for ep in cls.episodes}

    def test_quality_gate_audit(self):
        report = audit_dataset_quality(self.episodes)
        self.assertEqual(report.total_episodes, 12)
        self.assertEqual(report.valid_schema_count, 12)
        self.assertGreaterEqual(report.learnable_count, 10)

    def test_majority_baseline(self):
        baseline = MajorityBaseline()
        preds = {ep.episode_id: baseline.predict_episode(ep) for ep in self.episodes}
        metrics = Evaluator.evaluate(preds, self.ground_truth)
        self.assertIn("exact_match_accuracy", metrics)
        # Majority should have low exact match on balanced dataset
        self.assertLessEqual(metrics["exact_match_accuracy"], 50.0)

    def test_rule_based_learner(self):
        learner = RuleBasedLearner()
        preds = {ep.episode_id: learner.predict_episode(ep) for ep in self.episodes}
        metrics = Evaluator.evaluate(preds, self.ground_truth)
        # Rule based learner should achieve substantially higher accuracy than majority
        self.assertGreater(metrics["exact_match_accuracy"], 50.0)
        self.assertGreater(metrics["slot_accuracies"]["agent"], 60.0)

    def test_robustness_attacks(self):
        results = ShortcutAuditor.run_full_robustness_suite(self.episodes)
        self.assertIn("metadata_attack", results)
        self.assertIn("support_ablation_attack", results)
        self.assertTrue(results["support_ablation_attack"]["support_dependency_confirmed"])


if __name__ == "__main__":
    unittest.main()
