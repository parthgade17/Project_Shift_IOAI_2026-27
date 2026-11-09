"""
Unit tests for strengthened features:
- Factorial Semantic Disentanglement
- Statistical Naive Bayes Baseline
- Counterfactual Dialogue Pairs
"""

import unittest
from shift.ontology import ActionPredicate, ObjectEntity, Tense, DriftSource, EventFrame
from shift.drift import DriftEngine, CommunityId, HistoricalPeriod
from shift.episodes import EpisodeGenerator
from baselines.statistical import StatisticalLearner
from baselines.rule_based import RuleBasedLearner
from baselines.evaluate import Evaluator


class TestStrengthenedFeatures(unittest.TestCase):

    def test_factorial_drift_disentanglement(self):
        # 1. Community Main Effect: Northern Highlands has harvest -> 'kor'
        drift_north = DriftEngine(CommunityId.NORTHERN_HIGHLANDS.value, HistoricalPeriod.CLASSICAL.value)
        self.assertEqual(
            drift_north.get_drift_attribution(ActionPredicate.HARVEST.value),
            DriftSource.COMMUNITY_SHIFT.value
        )
        self.assertEqual(
            drift_north.get_drift_attribution(ActionPredicate.INSPECT.value),
            DriftSource.STABLE.value
        )

        # 2. Historical Main Effect: Archaic past tense across communities
        drift_archaic = DriftEngine(CommunityId.CITADEL.value, HistoricalPeriod.ARCHAIC.value)
        self.assertEqual(
            drift_archaic.get_drift_attribution(Tense.PAST.value),
            DriftSource.HISTORICAL_SHIFT.value
        )

        # 3. Factorial Interaction Effect: Coastal Valley in Contemporary Epoch
        drift_coast_contemp = DriftEngine(CommunityId.COASTAL_VALLEY.value, HistoricalPeriod.CONTEMPORARY.value)
        self.assertEqual(
            drift_coast_contemp.get_drift_attribution(ObjectEntity.CRYSTAL_CORE.value),
            DriftSource.INTERACTION_SHIFT.value
        )

    def test_counterfactual_pair_generation(self):
        gen = EpisodeGenerator(base_seed=888)
        ep_a, ep_b = gen.generate_counterfactual_episode_pair("pair_test_01")

        # Must have identical query utterance
        self.assertEqual(ep_a.query.utterance, ep_b.query.utterance)
        # Must have different dialogue contexts
        self.assertNotEqual(ep_a.query.context, ep_b.query.context)
        # Must have different ground truth interpretations!
        self.assertNotEqual(ep_a.query_label.agent, ep_b.query_label.agent)
        self.assertNotEqual(ep_a.query_label.object, ep_b.query_label.object)
        # Must share counterfactual pair ID
        self.assertEqual(ep_a.counterfactual_pair_id, "pair_test_01")
        self.assertEqual(ep_b.counterfactual_pair_id, "pair_test_01")

    def test_statistical_baseline(self):
        gen = EpisodeGenerator(base_seed=555)
        episodes = gen.generate_batch(count=12, k_support=8)
        gt = {ep.episode_id: ep.query_label for ep in episodes}

        learner = StatisticalLearner()
        preds = {ep.episode_id: learner.predict_episode(ep) for ep in episodes}
        metrics = Evaluator.evaluate(preds, gt)

        self.assertIn("exact_match_accuracy", metrics)
        self.assertGreater(metrics["slot_accuracies"]["tense"], 60.0)

    def test_rule_based_disentanglement_prediction(self):
        gen = EpisodeGenerator(base_seed=777)
        episodes = gen.generate_batch(count=8, k_support=8)
        gt = {ep.episode_id: ep.query_label for ep in episodes}

        learner = RuleBasedLearner()
        preds = {ep.episode_id: learner.predict_episode(ep) for ep in episodes}
        metrics = Evaluator.evaluate(preds, gt)

        # Learner should predict drift_attribution
        self.assertIn("disentanglement_attribution_accuracy", metrics)
        self.assertGreater(metrics["disentanglement_attribution_accuracy"], 50.0)


if __name__ == "__main__":
    unittest.main()
