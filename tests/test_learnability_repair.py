"""
Unit & Integration Tests for the Improved Learnability Validation and Repair System.
Tests:
- Check 1: Lexical Inferability
- Check 2: Grammar Inferability
- Check 3: Semantic Drift Inferability
- Check 4: Context Sufficiency
- Check 5: Unique Interpretation
- Check 6: Compositional Learnability
- Automatic In-Place Repair Engine (repair_episode)
- Dataset Audit & Guarantee Pipeline (audit_and_repair_dataset)
- Detailed Validation Report Generation (format_markdown, to_dict)
"""

import unittest
from typing import List, Dict, Any

from shift.ontology import (
    AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity, EventFrame, DriftSource
)
from shift.episodes import Episode, SupportExample, QueryInstance, EpisodeGenerator
from shift.learnability_validator import (
    LearnabilityValidator, CheckType, CheckResult,
    LearnabilityAuditResult, DatasetValidationReport
)


class TestLearnabilityValidationAndRepair(unittest.TestCase):
    def setUp(self):
        self.frame_valid = EventFrame(
            agent=AgentEntity.ARTISAN_GAMMA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value
        )
        self.support_valid = [
            SupportExample(
                example_id=1,
                utterance="artisan_gamma inspect-il energy_matrix",
                event_frame=self.frame_valid
            ),
            SupportExample(
                example_id=2,
                utterance="scout_alpha repair-il water_purifier",
                event_frame=EventFrame(
                    agent=AgentEntity.SCOUT_ALPHA.value,
                    action=ActionPredicate.REPAIR.value,
                    object=ObjectEntity.WATER_PURIFIER.value,
                    tense=Tense.PAST.value,
                    polarity=Polarity.AFFIRMATIVE.value
                )
            ),
            SupportExample(
                example_id=3,
                utterance="scholar_kappa activate-en crystal_core",
                event_frame=EventFrame(
                    agent=AgentEntity.SCHOLAR_KAPPA.value,
                    action=ActionPredicate.ACTIVATE.value,
                    object=ObjectEntity.CRYSTAL_CORE.value,
                    tense=Tense.PRESENT.value,
                    polarity=Polarity.AFFIRMATIVE.value
                )
            ),
            SupportExample(
                example_id=4,
                utterance="guardian_theta secure-or signal_beacon",
                event_frame=EventFrame(
                    agent=AgentEntity.GUARDIAN_THETA.value,
                    action=ActionPredicate.SECURE.value,
                    object=ObjectEntity.SIGNAL_BEACON.value,
                    tense=Tense.FUTURE.value,
                    polarity=Polarity.AFFIRMATIVE.value
                )
            )
        ]
        self.query_valid = QueryInstance(
            utterance="artisan_gamma inspect-il energy_matrix",
            context=[],
            phenomenon="direct"
        )
        self.valid_episode = Episode(
            episode_id="ep_audit_01",
            community_id="comm_citadel",
            period_id="period_classical",
            word_order="SVO",
            support_set=self.support_valid,
            query=self.query_valid,
            query_label=self.frame_valid,
            drift_attribution={"action": "stable"}
        )

    # ---------------------------------------------------------
    # Test Check 1: Lexical Inferability
    # ---------------------------------------------------------
    def test_lexical_inferability_pass(self):
        result = LearnabilityValidator.check_lexical_inferability(self.valid_episode)
        self.assertTrue(result.passed)
        self.assertEqual(len(result.reasons), 0)

    def test_lexical_inferability_fail_missing_agent(self):
        # Query uses scout_beta, not present in support set
        bad_frame = EventFrame(
            agent=AgentEntity.SCOUT_BETA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value
        )
        bad_ep = Episode(
            episode_id="bad_ep_lex",
            community_id="comm_citadel",
            period_id="period_classical",
            word_order="SVO",
            support_set=self.support_valid,
            query=QueryInstance(utterance="scout_beta inspect-il energy_matrix", phenomenon="direct"),
            query_label=bad_frame
        )
        result = LearnabilityValidator.check_lexical_inferability(bad_ep)
        self.assertFalse(result.passed)
        self.assertTrue(any("scout_beta" in r for r in result.reasons))

    # ---------------------------------------------------------
    # Test Check 2: Grammar Inferability
    # ---------------------------------------------------------
    def test_grammar_inferability_pass(self):
        result = LearnabilityValidator.check_grammar_inferability(self.valid_episode)
        self.assertTrue(result.passed)

    def test_grammar_inferability_fail_missing_tense(self):
        # Support only has past and present, but query is future
        support_no_future = [self.support_valid[0], self.support_valid[1], self.support_valid[2]]
        future_frame = EventFrame(
            agent=AgentEntity.ARTISAN_GAMMA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.FUTURE.value,
            polarity=Polarity.AFFIRMATIVE.value
        )
        ep = Episode(
            episode_id="ep_tense_fail",
            community_id="comm_citadel",
            period_id="period_classical",
            word_order="SVO",
            support_set=support_no_future,
            query=QueryInstance(utterance="artisan_gamma inspect-or energy_matrix", phenomenon="direct"),
            query_label=future_frame
        )
        result = LearnabilityValidator.check_grammar_inferability(ep)
        self.assertFalse(result.passed)
        self.assertTrue(any("future" in r.lower() for r in result.reasons))

    # ---------------------------------------------------------
    # Test Check 3: Semantic Drift Inferability
    # ---------------------------------------------------------
    def test_semantic_drift_inferability(self):
        drift_frame = EventFrame(
            agent=AgentEntity.ARTISAN_GAMMA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value,
            drift_attribution={"action": "community_shift"}
        )
        ep = Episode(
            episode_id="ep_drift",
            community_id="comm_citadel",
            period_id="period_classical",
            word_order="SVO",
            support_set=self.support_valid,
            query=self.query_valid,
            query_label=drift_frame,
            drift_attribution={"action": "community_shift"}
        )
        # Action 'inspect' is in support set, so drift inferability passes
        result = LearnabilityValidator.check_semantic_drift_inferability(ep)
        self.assertTrue(result.passed)

    # ---------------------------------------------------------
    # Test Check 4: Context Sufficiency
    # ---------------------------------------------------------
    def test_context_sufficiency_fail_empty_context(self):
        ep = Episode(
            episode_id="ep_ctx_fail",
            community_id="comm_citadel",
            period_id="period_classical",
            word_order="SVO",
            support_set=self.support_valid,
            query=QueryInstance(utterance="la id inspect-il", context=[], phenomenon="coreference"),
            query_label=self.frame_valid
        )
        result = LearnabilityValidator.check_context_sufficiency(ep)
        self.assertFalse(result.passed)
        self.assertTrue(any("empty" in r.lower() for r in result.reasons))

    def test_context_sufficiency_pass_with_context(self):
        ctx_turn = {
            "turn_id": 1,
            "speaker": "Speaker A",
            "utterance": "artisan_gamma inspect-il energy_matrix",
            "event_frame": self.frame_valid.to_dict(),
            "reference_type_used": "direct"
        }
        ep = Episode(
            episode_id="ep_ctx_ok",
            community_id="comm_citadel",
            period_id="period_classical",
            word_order="SVO",
            support_set=self.support_valid,
            query=QueryInstance(utterance="la id inspect-il", context=[ctx_turn], phenomenon="coreference"),
            query_label=self.frame_valid
        )
        result = LearnabilityValidator.check_context_sufficiency(ep)
        self.assertTrue(result.passed)

    # ---------------------------------------------------------
    # Test Check 5: Unique Interpretation
    # ---------------------------------------------------------
    def test_unique_interpretation_ambiguity_detected(self):
        # Two competing feminine agents in context: Scout Beta and Scholar Kappa
        ctx = [
            {"turn_id": 1, "speaker": "A", "event_frame": {"agent": AgentEntity.SCOUT_BETA.value, "gender": "feminine"}},
            {"turn_id": 2, "speaker": "B", "event_frame": {"agent": AgentEntity.SCHOLAR_KAPPA.value, "gender": "feminine"}}
        ]
        target = EventFrame(
            agent=AgentEntity.SCOUT_BETA.value, # Not the last one!
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value
        )
        ep = Episode(
            episode_id="ep_ambig",
            community_id="comm_citadel",
            period_id="period_classical",
            word_order="SVO",
            support_set=self.support_valid,
            query=QueryInstance(utterance="la inspect-il energy_matrix", context=ctx, phenomenon="coreference"),
            query_label=target
        )
        result = LearnabilityValidator.check_unique_interpretation(ep)
        self.assertFalse(result.passed)
        self.assertTrue(any("ambiguous" in r.lower() for r in result.reasons))

    # ---------------------------------------------------------
    # Test Check 6: Compositional Learnability
    # ---------------------------------------------------------
    def test_compositional_learnability(self):
        result = LearnabilityValidator.check_compositional_learnability(self.valid_episode)
        self.assertTrue(result.passed)

    # ---------------------------------------------------------
    # Test Automatic In-Place Repair Engine
    # ---------------------------------------------------------
    def test_repair_missing_lexical_agent(self):
        # Query needs scout_beta, which is missing from support set
        target = EventFrame(
            agent=AgentEntity.SCOUT_BETA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value
        )
        ep = Episode(
            episode_id="ep_to_repair",
            community_id="comm_citadel",
            period_id="period_classical",
            word_order="SVO",
            support_set=self.support_valid,
            query=QueryInstance(utterance="scout_beta inspect-il energy_matrix", context=[], phenomenon="direct"),
            query_label=target
        )
        # Initially fails
        init_audit = LearnabilityValidator.audit_episode(ep)
        self.assertFalse(init_audit.is_solvable)

        # In-place repair
        repaired_ep, success, actions = LearnabilityValidator.repair_episode(ep)
        self.assertTrue(success)
        self.assertTrue(len(actions) > 0)

        # Post-repair audit must pass 100%
        post_audit = LearnabilityValidator.audit_episode(repaired_ep)
        self.assertTrue(post_audit.is_solvable)
        self.assertTrue(post_audit.lexical_inferability)

    # ---------------------------------------------------------
    # Test Dataset Validation Report
    # ---------------------------------------------------------
    def test_audit_and_repair_dataset_yields_100_percent(self):
        generator = EpisodeGenerator(base_seed=42)
        batch = generator.generate_batch(count=10, k_support=8)

        certified, report = LearnabilityValidator.audit_and_repair_dataset(batch, generator=generator)
        self.assertEqual(len(certified), 10)
        self.assertEqual(report.pass_rate, 100.0)
        self.assertEqual(report.learnable_count, 10)

        md = report.format_markdown()
        self.assertIn("Six Foundational Learnability Checks Breakdown", md)
        self.assertIn("100.0%", md)


if __name__ == "__main__":
    unittest.main()
