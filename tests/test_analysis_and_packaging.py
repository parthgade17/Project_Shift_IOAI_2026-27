"""
Project SHIFT: Automated Tests for Analysis and Packaging Suite.
IOAI 2027 Scientific Benchmark.
"""

import sys
import unittest
import tempfile
import zipfile
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.error_analysis import run_error_analysis
from analysis.model_comparison import run_model_comparison
from scripts.profile_ioai_limits import profile_ioai_limits
from scripts.package_release import package_releases


class TestAnalysisAndPackaging(unittest.TestCase):
    """Verifies error taxonomy analysis, model comparison export, resource profiling, and release packaging."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_error_analysis_execution(self):
        """Tests that error analysis runs and exports diagnostic JSON."""
        res = run_error_analysis(
            test_path="datasets/hidden_test_full_reference.jsonl",
            output_dir=self.output_dir
        )
        self.assertEqual(res["analysis_type"], "diagnostic_failure_mode_taxonomy")
        self.assertIn("Inductive Rule Learner", res["models"])
        self.assertTrue((self.output_dir / "error_analysis.json").exists())

    def test_model_comparison_exports(self):
        """Tests that model comparison generates Markdown, LaTeX, and JSON tables."""
        res = run_model_comparison(
            test_path="datasets/hidden_test_full_reference.jsonl",
            output_dir=self.output_dir
        )
        self.assertEqual(res["analysis_type"], "cross_model_comparison")
        self.assertTrue((self.output_dir / "model_comparison.json").exists())
        self.assertTrue((self.output_dir / "model_comparison.md").exists())
        self.assertTrue((self.output_dir / "model_comparison.tex").exists())

    def test_profile_ioai_limits(self):
        """Tests that resource profiler measures latency, memory, and verifies IOAI constraints."""
        out_file = self.output_dir / "compliance.json"
        res = profile_ioai_limits(
            dataset_path="datasets/hidden_test_full_reference.jsonl",
            output_path=str(out_file)
        )
        self.assertTrue(res["all_passed"])
        self.assertTrue(out_file.exists())
        self.assertIn("runtime_compliance", res["verdicts"])
        self.assertTrue(res["verdicts"]["runtime_compliance"]["passed"])

    def test_package_releases_cleanliness(self):
        """Tests that release packaging builds clean contestant and author zips without leakage."""
        c_zip, a_zip = package_releases()
        self.assertTrue(c_zip.exists())
        self.assertTrue(a_zip.exists())

        # Audit Contestant Zip
        with zipfile.ZipFile(c_zip, "r") as zf:
            names = zf.namelist()
            # Must include task statement, loader, template, evaluator, dev data
            self.assertTrue(any("IOAI_TASK_STATEMENT.md" in n for n in names))
            self.assertTrue(any("loader.py" in n for n in names))
            self.assertTrue(any("submission_template.py" in n for n in names))
            self.assertTrue(any("local_evaluator.py" in n for n in names))

            # Must NOT include private keys
            for n in names:
                lower = n.lower()
                self.assertNotIn("hidden_test_labels", lower)
                self.assertNotIn("hidden_test_full_reference", lower)
                self.assertNotIn("__pycache__", lower)
                self.assertNotIn(".pyc", lower)

        # Audit Author Zip
        with zipfile.ZipFile(a_zip, "r") as zf:
            names = zf.namelist()
            self.assertTrue(any("grade_submission.py" in n for n in names))
            self.assertTrue(any("hidden_test_labels.json" in n for n in names))
            self.assertTrue(any("hidden_test_full_reference.jsonl" in n for n in names))


if __name__ == "__main__":
    unittest.main()
