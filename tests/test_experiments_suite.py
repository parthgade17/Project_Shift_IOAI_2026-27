"""
Project SHIFT: Automated Tests for Experiment Suite (EXP-01 to EXP-07).
IOAI 2027 Scientific Benchmark.
"""

import sys
import unittest
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.support_size import run_support_size_experiment
from experiments.context_ablation import run_context_ablation_experiment
from experiments.community_generalisation import run_community_generalisation_experiment
from experiments.historical_generalisation import run_historical_generalisation_experiment
from experiments.composition_test import run_composition_test_experiment
from experiments.counterfactual_test import run_counterfactual_test_experiment
from experiments.robustness_test import run_robustness_test_experiment


class TestExperimentSuite(unittest.TestCase):
    """Verifies that all modular experiments execute cleanly and return valid schema dictionaries."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exp01_support_size_runs(self):
        """Tests that EXP-01 runs with mini config and returns required keys."""
        res = run_support_size_experiment(
            k_values=[0, 4],
            seeds=[42],
            count_per_seed=4,
            output_dir=self.output_dir
        )
        self.assertEqual(res["experiment_id"], "EXP-01")
        self.assertIn("k_0", res["results"])
        self.assertIn("k_4", res["results"])
        self.assertTrue((self.output_dir / "support_size_curve.json").exists())

    def test_exp02_context_ablation_runs(self):
        """Tests that EXP-02 computes full vs ablated context drop."""
        res = run_context_ablation_experiment(
            dataset_path="datasets/hidden_test_full_reference.jsonl",
            output_dir=self.output_dir
        )
        self.assertEqual(res["experiment_id"], "EXP-02")
        self.assertIn("results_by_model", res)
        self.assertIn("Inductive Rule Learner", res["results_by_model"])
        rule_m = res["results_by_model"]["Inductive Rule Learner"]
        self.assertIn("ablation_drop_no_context", rule_m)
        self.assertTrue((self.output_dir / "context_ablation.json").exists())

    def test_exp03_community_generalisation_runs(self):
        """Tests that EXP-03 computes seen vs held-out community gap."""
        res = run_community_generalisation_experiment(
            iid_path="datasets/test_iid.jsonl",
            heldout_path="datasets/test_heldout_community.jsonl",
            output_dir=self.output_dir
        )
        self.assertEqual(res["experiment_id"], "EXP-03")
        self.assertIn("Inductive Rule Learner", res["results"])
        self.assertTrue((self.output_dir / "community_generalisation.json").exists())

    def test_exp04_historical_generalisation_runs(self):
        """Tests that EXP-04 computes seen vs held-out historical epoch gap."""
        res = run_historical_generalisation_experiment(
            iid_path="datasets/test_iid.jsonl",
            heldout_path="datasets/test_heldout_period.jsonl",
            output_dir=self.output_dir
        )
        self.assertEqual(res["experiment_id"], "EXP-04")
        self.assertIn("Inductive Rule Learner", res["results"])
        self.assertTrue((self.output_dir / "historical_generalisation.json").exists())

    def test_exp05_composition_test_runs(self):
        """Tests that EXP-05 evaluates Level 5 compositional OOD."""
        res = run_composition_test_experiment(
            iid_path="datasets/test_iid.jsonl",
            ood_path="datasets/test_ood.jsonl",
            output_dir=self.output_dir
        )
        self.assertEqual(res["experiment_id"], "EXP-05")
        self.assertIn("Inductive Rule Learner", res["results"])
        self.assertIn("action_object_compound_accuracy", res["results"]["Inductive Rule Learner"]["ood_compositional"])
        self.assertTrue((self.output_dir / "compositional_ood.json").exists())

    def test_exp06_counterfactual_test_runs(self):
        """Tests that EXP-06 measures joint accuracy and perturbation sensitivity."""
        res = run_counterfactual_test_experiment(
            cf_path="datasets/test_counterfactual.jsonl",
            output_dir=self.output_dir
        )
        self.assertEqual(res["experiment_id"], "EXP-06")
        self.assertIn("Inductive Rule Learner", res["results"])
        cf_m = res["results"]["Inductive Rule Learner"]
        self.assertIn("joint_accuracy", cf_m)
        self.assertIn("perturbation_sensitivity", cf_m)
        self.assertTrue((self.output_dir / "counterfactual_results.json").exists())

    def test_exp07_robustness_test_runs(self):
        """Tests that EXP-07 computes word order, polarity, and morphology breakdowns."""
        res = run_robustness_test_experiment(
            test_path="datasets/hidden_test_full_reference.jsonl",
            output_dir=self.output_dir
        )
        self.assertEqual(res["experiment_id"], "EXP-07")
        self.assertIn("Inductive Rule Learner", res["results"])
        self.assertIn("word_order_breakdown", res["results"]["Inductive Rule Learner"])
        self.assertTrue((self.output_dir / "robustness_results.json").exists())


if __name__ == "__main__":
    unittest.main()
