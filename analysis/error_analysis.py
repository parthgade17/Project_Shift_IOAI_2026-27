"""
Project SHIFT: Automated Diagnostic Error Analysis.
IOAI 2027 Scientific Benchmark Analysis Suite.

Analyzes model failure modes across the 5 diagnostic categories:
  1. LexicalInductionFailure: Misaligned root entity/action morphemes.
  2. MorphosyntacticFailure: Misparsed tense or polarity affixes.
  3. DiscourseTrackingFailure: Misresolved coreference, ellipsis, or repair.
  4. DisentanglementFailure: Correct semantics but failed causal drift attribution.
  5. CounterfactualContextBias: Rigid output invariant to context perturbation.

Computes failure frequency breakdowns, per-slot confusion rates, and pedagogical explanations.
"""

from __future__ import annotations
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shift.serialize import load_episodes_jsonl
from shift.diagnostics import ModelDiagnostics
from baselines.majority import MajorityBaseline
from baselines.surface_lookup import SurfaceLookupBaseline
from baselines.statistical import StatisticalLearner
from baselines.rule_based import RuleBasedLearner


def run_error_analysis(
    test_path: str = "datasets/hidden_test_full_reference.jsonl",
    output_dir: Path = None,
) -> Dict[str, Any]:
    if output_dir is None:
        output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(" PROJECT SHIFT: DIAGNOSTIC ERROR ANALYSIS")
    print(f" Source dataset: {test_path}")
    print("=" * 72)

    episodes = load_episodes_jsonl(test_path)
    print(f"Loaded {len(episodes)} test episodes for diagnostic audit.")

    baselines = [
        ("Majority Baseline", MajorityBaseline()),
        ("Surface Lookup Baseline", SurfaceLookupBaseline()),
        ("Statistical Naive Bayes", StatisticalLearner()),
        ("Inductive Rule Learner", RuleBasedLearner()),
    ]

    diagnostics_by_model: Dict[str, Any] = {}

    for name, model in baselines:
        print(f"\n--> Running Error Diagnostics for: {name}")
        preds = {ep.episode_id: model.predict_episode(ep) for ep in episodes}
        report = ModelDiagnostics.diagnose_all(episodes, preds)

        diagnostics_by_model[name] = report.to_dict()

        print(f"    * Total Errors           : {report.total_errors} / {report.total_episodes} ({100.0 - report.exact_match_accuracy:.1f}%)")
        print("    * Failure Modes Breakdown:")
        for mode, count in report.failure_counts.items():
            pct = report.failure_percentages.get(mode, 0.0)
            print(f"      - {mode:<26}: {count:3d} errors ({pct:5.1f}% of total errors)")

    out_file = output_dir / "error_analysis.json"
    summary_output = {
        "analysis_type": "diagnostic_failure_mode_taxonomy",
        "benchmark": "Project SHIFT (IOAI 2027)",
        "num_episodes": len(episodes),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": diagnostics_by_model,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_output, f, indent=2)
    print(f"\n[OK] Diagnostic error analysis saved to {out_file}")

    return summary_output


if __name__ == "__main__":
    run_error_analysis()
