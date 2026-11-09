"""
Unit Tests for Project SHIFT v0.3 Scientific Benchmark Infrastructure.
Tests:
- shift.schema (SchemaValidator)
- shift.learnability_validator (LearnabilityValidator)
- shift.shortcut_audits (ShortcutAuditor)
- shift.metrics (compute_all_metrics, compute_slot_f1)
- shift.evaluator (Evaluator)
- shift.diagnostics (ModelDiagnostics, FailureMode)
- shift.evaluation_report (ReportGenerator)
"""

import unittest
import tempfile
import json
from pathlib import Path

from shift.ontology import (
    AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity, DriftSource, EventFrame
)
from shift.episodes import Episode, SupportExample, QueryInstance
from shift.schema import SchemaValidator, SchemaValidationError, get_event_frame_schema
from shift.learnability_validator import LearnabilityValidator, LearnabilityAuditResult
from shift.shortcut_audits import ShortcutAuditor
from shift.metrics import compute_all_metrics, compute_slot_f1
from shift.evaluator import Evaluator
from shift.diagnostics import ModelDiagnostics, FailureMode
from shift.evaluation_report import ReportGenerator
from baselines.majority import MajorityBaseline


class TestSchemaValidator(unittest.TestCase):
    def setUp(self):
        self.valid_event = {
            "agent": "artisan_gamma",
            "action": "inspect",
            "object": "energy_matrix",
            "tense": "past",
            "polarity": "affirmative",
            "resolved_references": {"la": "artisan_gamma"},
            "drift_attribution": {
                "agent": "stable",
                "action": "community_shift"
            }
        }

    def test_valid_event(self):
        is_valid, errors = SchemaValidator.validate_event_dict(self.valid_event)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

    def test_missing_required_field(self):
        bad_event = dict(self.valid_event)
        del bad_event["action"]
        is_valid, errors = SchemaValidator.validate_event_dict(bad_event)
        self.assertFalse(is_valid)
        self.assertTrue(any(e.field == "action" for e in errors))

    def test_invalid_enum_value(self):
        bad_event = dict(self.valid_event)
        bad_event["agent"] = "invalid_alien_creature"
        is_valid, errors = SchemaValidator.validate_event_dict(bad_event)
        self.assertFalse(is_valid)
        self.assertTrue(any(e.field == "agent" for e in errors))

    def test_invalid_drift_attribution(self):
        bad_event = dict(self.valid_event)
        bad_event["drift_attribution"] = {"action": "magic_shift"}
        is_valid, errors = SchemaValidator.validate_event_dict(bad_event)
        self.assertFalse(is_valid)
        self.assertTrue(any("drift_attribution" in e.field for e in errors))

    def test_get_event_frame_schema(self):
        schema = get_event_frame_schema()
        self.assertIn("properties", schema)
        self.assertIn("agent", schema["properties"])
        self.assertEqual(schema["type"], "object")


class TestLearnabilityValidator(unittest.TestCase):
    def setUp(self):
        self.support = [
            SupportExample(
                example_id=1,
                utterance="artisan_gamma inspect-il energy_matrix",
                event_frame=EventFrame(
                    agent=AgentEntity.ARTISAN_GAMMA.value,
                    action=ActionPredicate.INSPECT.value,
                    object=ObjectEntity.ENERGY_MATRIX.value,
                    tense=Tense.PAST.value,
                    polarity=Polarity.AFFIRMATIVE.value
                )
            )
        ]
        self.query = QueryInstance(
            utterance="artisan_gamma inspect-il energy_matrix",
            phenomenon="direct"
        )
        self.valid_episode = Episode(
            episode_id="ep_test_01",
            community_id="Harbor",
            period_id="Modern",
            word_order="SVO",
            support_set=self.support,
            query=self.query,
            query_label=self.support[0].event_frame
        )

    def test_learnable_episode(self):
        audit = LearnabilityValidator.audit_episode(self.valid_episode)
        self.assertTrue(audit.is_solvable)
        self.assertTrue(audit.lexical_grounding)

    def test_unlearnable_missing_lexicon(self):
        bad_query = QueryInstance(
            utterance="scholar_kappa inspect-il energy_matrix",
            phenomenon="direct"
        )
        bad_episode = Episode(
            episode_id="ep_test_02",
            community_id="Harbor",
            period_id="Modern",
            word_order="SVO",
            support_set=self.support,
            query=bad_query,
            query_label=EventFrame(
                agent=AgentEntity.SCHOLAR_KAPPA.value,
                action=ActionPredicate.INSPECT.value,
                object=ObjectEntity.ENERGY_MATRIX.value,
                tense=Tense.PAST.value,
                polarity=Polarity.AFFIRMATIVE.value
            )
        )
        audit = LearnabilityValidator.audit_episode(bad_episode)
        self.assertFalse(audit.lexical_grounding)
        self.assertFalse(audit.is_solvable)


class TestShortcutAuditor(unittest.TestCase):
    def setUp(self):
        self.frame = EventFrame(
            agent=AgentEntity.ARTISAN_GAMMA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value
        )
        self.ep = Episode(
            episode_id="audit_ep_1",
            community_id="Forest",
            period_id="Ancient",
            word_order="SVO",
            support_set=[
                SupportExample(
                    example_id=1,
                    utterance="artisan_gamma inspect energy_matrix",
                    event_frame=self.frame
                )
            ],
            query=QueryInstance(
                utterance="artisan_gamma inspect energy_matrix",
                context=[
                    {"speaker": "A", "utterance": "artisan_gamma inspect energy_matrix"},
                    {"speaker": "B", "utterance": "yes indeed"}
                ],
                phenomenon="direct"
            ),
            query_label=self.frame
        )

    def test_metadata_audit(self):
        res = ShortcutAuditor.audit_metadata_shortcut([self.ep])
        self.assertIn("attack_type", res)
        self.assertIn("passed", res)

    def test_support_ablation_audit(self):
        res = ShortcutAuditor.audit_support_ablation([self.ep])
        self.assertIn("attack_type", res)
        self.assertIn("support_dependency_confirmed", res)

    def test_full_shortcut_suite(self):
        res = ShortcutAuditor.run_full_shortcut_suite([self.ep])
        self.assertIn("all_shortcut_audits_passed", res)
        self.assertIn("metadata_attack", res)
        self.assertIn("support_ablation", res)


class TestMetricsAndEvaluator(unittest.TestCase):
    def setUp(self):
        self.f1 = EventFrame(
            agent=AgentEntity.ARTISAN_GAMMA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value,
            resolved_references={"la": "artisan_gamma"},
            drift_attribution={"agent": DriftSource.STABLE.value}
        )
        self.f2 = EventFrame(
            agent=AgentEntity.SCHOLAR_KAPPA.value,
            action=ActionPredicate.REPAIR.value,
            object=ObjectEntity.WATER_PURIFIER.value,
            tense=Tense.PRESENT.value,
            polarity=Polarity.NEGATIVE.value,
            resolved_references={},
            drift_attribution={"agent": DriftSource.COMMUNITY_SHIFT.value}
        )
        self.gt = {"ep_1": self.f1, "ep_2": self.f2}

    def test_compute_slot_f1(self):
        preds = {"ep_1": self.f1, "ep_2": self.f1} # 2nd prediction wrong
        f1_res = compute_slot_f1(preds, self.gt, "agent")
        self.assertIn("macro_f1", f1_res)
        self.assertGreaterEqual(f1_res["macro_f1"], 0.0)

    def test_evaluator_evaluate(self):
        preds = {"ep_1": self.f1.to_dict(), "ep_2": self.f2.to_dict()}
        res = Evaluator.evaluate(preds, self.gt)
        self.assertEqual(res["exact_match_accuracy"], 100.0)
        self.assertEqual(res["composite_competition_score"], 100.0)
        self.assertTrue(res["schema_validation"]["is_perfectly_formed"])

    def test_evaluator_file_evaluation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            preds_file = Path(tmpdir) / "preds.json"
            labels_file = Path(tmpdir) / "labels.json"
            out_file = Path(tmpdir) / "out.json"

            with open(preds_file, "w", encoding="utf-8") as f:
                json.dump({"ep_1": self.f1.to_dict()}, f)
            with open(labels_file, "w", encoding="utf-8") as f:
                json.dump({"ep_1": self.f1.to_dict()}, f)

            res = Evaluator.evaluate_files(
                predictions_path=preds_file,
                labels_path=labels_file,
                output_path=out_file
            )
            self.assertEqual(res["exact_match_accuracy"], 100.0)
            self.assertTrue(out_file.exists())


class TestModelDiagnostics(unittest.TestCase):
    def setUp(self):
        self.gt = EventFrame(
            agent=AgentEntity.ARTISAN_GAMMA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value,
            resolved_references={"la": "artisan_gamma"},
            drift_attribution={"action": DriftSource.COMMUNITY_SHIFT.value}
        )
        self.ep = Episode(
            episode_id="diag_ep_1",
            community_id="Valley",
            period_id="Ancient",
            word_order="SOV",
            support_set=[],
            query=QueryInstance(utterance="la inspect-il energy_matrix", phenomenon="coreference"),
            query_label=self.gt,
            difficulty_level=3
        )

    def test_diagnose_lexical_failure(self):
        pred = EventFrame(
            agent=AgentEntity.SCHOLAR_KAPPA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value,
            resolved_references={"la": "artisan_gamma"}
        )
        diag = ModelDiagnostics.diagnose_episode(self.ep, pred)
        self.assertFalse(diag.is_exact_match)
        self.assertIn(FailureMode.LEXICAL_INDUCTION, diag.failure_modes)

    def test_diagnose_morphosyntactic_failure(self):
        pred = EventFrame(
            agent=AgentEntity.ARTISAN_GAMMA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.FUTURE.value, # Wrong tense
            polarity=Polarity.AFFIRMATIVE.value,
            resolved_references={"la": "artisan_gamma"}
        )
        diag = ModelDiagnostics.diagnose_episode(self.ep, pred)
        self.assertIn(FailureMode.MORPHOSYNTACTIC, diag.failure_modes)

    def test_diagnose_disentanglement_failure(self):
        pred = EventFrame(
            agent=AgentEntity.ARTISAN_GAMMA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value,
            resolved_references={"la": "artisan_gamma"},
            drift_attribution={"action": DriftSource.HISTORICAL_SHIFT.value} # Wrong source
        )
        diag = ModelDiagnostics.diagnose_episode(self.ep, pred)
        self.assertIn(FailureMode.DISENTANGLEMENT, diag.failure_modes)

    def test_diagnose_all_generates_report(self):
        pred = EventFrame(
            agent=AgentEntity.ARTISAN_GAMMA.value,
            action=ActionPredicate.INSPECT.value,
            object=ObjectEntity.ENERGY_MATRIX.value,
            tense=Tense.PAST.value,
            polarity=Polarity.AFFIRMATIVE.value,
            resolved_references={"la": "artisan_gamma"},
            drift_attribution={"action": DriftSource.COMMUNITY_SHIFT.value}
        )
        report = ModelDiagnostics.diagnose_all([self.ep], {self.ep.episode_id: pred})
        self.assertEqual(report.total_episodes, 1)
        self.assertEqual(report.exact_match_accuracy, 100.0)
        self.assertTrue(len(report.format_text_summary()) > 50)


class TestReportGenerator(unittest.TestCase):
    def test_report_generation(self):
        gen = ReportGenerator(title="Test IOAI Report")
        eval_res = {
            "exact_match_accuracy": 75.0,
            "composite_competition_score": 82.5,
            "disentanglement_accuracy": 70.0,
            "reference_resolution_accuracy": 85.0,
            "counterfactual_consistency": {"consistency_score": 90.0},
            "slot_macro_f1": {
                "agent": {"macro_f1": 80.0},
                "action": {"macro_f1": 85.0},
                "object": {"macro_f1": 75.0},
                "tense": {"macro_f1": 70.0},
                "polarity": {"macro_f1": 95.0}
            },
            "progressive_levels": {
                "Level_1": {"accuracy": 95.0},
                "Level_2": {"accuracy": 80.0},
                "Level_3": {"accuracy": 70.0},
                "Level_4": {"accuracy": 65.0},
                "Level_5": {"accuracy": 60.0}
            }
        }
        gen.add_model_results("InductiveRuleLearner", eval_res)
        md = gen.generate_markdown()
        latex = gen.generate_latex()

        self.assertIn("# Test IOAI Report", md)
        self.assertIn("InductiveRuleLearner", md)
        self.assertIn(r"\begin{table*}", latex)
        self.assertIn("InductiveRuleLearner", latex)


if __name__ == "__main__":
    unittest.main()
