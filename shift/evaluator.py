"""
Project SHIFT: Standard Prediction Evaluator.
Component 6 of SHIFT v0.3 Scientific Benchmark Infrastructure.

Provides standard, deterministic evaluation for competition submissions,
incorporating strict schema validation, slot-level F1, disentanglement accuracy,
counterfactual consistency, and progressive level analysis.
"""

from __future__ import annotations
import sys
import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from shift.ontology import EventFrame
from shift.episodes import Episode
from shift.schema import SchemaValidator, SchemaValidationError
from shift.metrics import compute_all_metrics
from shift.serialize import load_episodes_jsonl, load_ground_truth_labels


class Evaluator:
    """Standard, deterministic evaluation engine for Project SHIFT."""

    @classmethod
    def validate_and_parse_predictions(
        cls,
        raw_predictions: Dict[str, Any],
        strict: bool = False
    ) -> Tuple[Dict[str, EventFrame], Dict[str, List[SchemaValidationError]]]:
        """
        Validates raw prediction dictionaries against SchemaValidator and
        converts valid predictions to EventFrame instances.
        """
        parsed: Dict[str, EventFrame] = {}
        errors: Dict[str, List[SchemaValidationError]] = {}

        for ep_id, pred_dict in raw_predictions.items():
            if isinstance(pred_dict, EventFrame):
                parsed[ep_id] = pred_dict
                continue

            is_valid, validation_errors = SchemaValidator.validate_event_dict(pred_dict)
            if is_valid:
                try:
                    parsed[ep_id] = EventFrame.from_dict(pred_dict)
                except Exception as e:
                    errors[ep_id] = [SchemaValidationError(
                        field="root",
                        message=f"Deserialization error: {e}",
                        received_value=str(pred_dict)
                    )]
            else:
                errors[ep_id] = validation_errors

        if strict and errors:
            first_id = next(iter(errors))
            first_err = errors[first_id][0]
            raise ValueError(
                f"Strict evaluation aborted: Episode '{first_id}' failed schema validation: {first_err}"
            )

        return parsed, errors

    @classmethod
    def evaluate(
        cls,
        predictions: Dict[str, Any],
        ground_truth: Dict[str, Any],
        episodes_by_id: Optional[Dict[str, Episode]] = None,
        strict_schema: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates predictions against ground truth.
        Supports both raw dictionaries and EventFrame objects.
        """
        # Parse ground truth
        parsed_gt: Dict[str, EventFrame] = {}
        for ep_id, gt in ground_truth.items():
            if isinstance(gt, EventFrame):
                parsed_gt[ep_id] = gt
            elif isinstance(gt, dict):
                parsed_gt[ep_id] = EventFrame.from_dict(gt)
            else:
                raise TypeError(f"Invalid ground truth type for {ep_id}: {type(gt)}")

        # Parse & validate predictions
        parsed_preds, schema_errors = cls.validate_and_parse_predictions(
            predictions, strict=strict_schema
        )

        # Compute benchmark metrics
        metrics = compute_all_metrics(parsed_preds, parsed_gt, episodes_by_id)

        # Attach schema validation stats
        metrics["schema_validation"] = {
            "total_submitted": len(predictions),
            "valid_submissions": len(parsed_preds),
            "invalid_submissions": len(schema_errors),
            "is_perfectly_formed": len(schema_errors) == 0,
            "errors": {
                ep_id: [err.to_dict() for err in err_list]
                for ep_id, err_list in schema_errors.items()
            }
        }

        return metrics

    @classmethod
    def evaluate_files(
        cls,
        predictions_path: str | Path,
        dataset_path: Optional[str | Path] = None,
        labels_path: Optional[str | Path] = None,
        output_path: Optional[str | Path] = None,
        strict_schema: bool = False
    ) -> Dict[str, Any]:
        """Loads files and evaluates predictions."""
        pred_p = Path(predictions_path)
        if not pred_p.exists():
            raise FileNotFoundError(f"Predictions file not found: {pred_p}")

        # Load predictions
        with open(pred_p, "r", encoding="utf-8-sig") as f:
            if pred_p.suffix == ".jsonl":
                raw_preds = {}
                for line in f:
                    line = line.strip()
                    if line:
                        d = json.loads(line)
                        raw_preds[d["episode_id"]] = d.get("prediction", d)
            else:
                raw_preds = json.load(f)

        # Load ground truth and episodes
        episodes_by_id: Optional[Dict[str, Episode]] = None
        ground_truth: Dict[str, EventFrame] = {}

        if labels_path:
            lab_p = Path(labels_path)
            ground_truth = load_ground_truth_labels(lab_p)

        if dataset_path:
            ds_p = Path(dataset_path)
            episodes = load_episodes_jsonl(ds_p)
            episodes_by_id = {ep.episode_id: ep for ep in episodes}
            if not ground_truth:
                for ep in episodes:
                    if ep.query_label:
                        ground_truth[ep.episode_id] = ep.query_label

        if not ground_truth:
            raise ValueError(
                "No ground truth available. Provide either --labels <labels.json> or a --dataset containing query_label."
            )

        results = cls.evaluate(
            predictions=raw_preds,
            ground_truth=ground_truth,
            episodes_by_id=episodes_by_id,
            strict_schema=strict_schema
        )

        if output_path:
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)

        return results

    @classmethod
    def print_summary(cls, results: Dict[str, Any], title: str = "BENCHMARK EVALUATION RESULTS") -> None:
        """Prints a clean ASCII summary table to stdout."""
        print("\n" + "=" * 65)
        print(f" {title}")
        print("=" * 65)

        total = results.get("total_episodes", 0)
        em = results.get("exact_match_accuracy", 0.0)
        comp = results.get("composite_competition_score", 0.0)
        dis = results.get("disentanglement_accuracy", 0.0)
        ref = results.get("reference_resolution_accuracy", 0.0)

        schema = results.get("schema_validation", {})
        valid = schema.get("valid_submissions", 0)
        invalid = schema.get("invalid_submissions", 0)

        print(f"Total Episodes Evaluated : {total}")
        print(f"Schema Valid Predictions : {valid}/{total} ({'PERFECT' if invalid == 0 else f'{invalid} ERRORS'})")
        print("-" * 65)
        print(f"Exact Match Accuracy     : {em:6.2f}%")
        print(f"Composite Score (IOAI)   : {comp:6.2f} / 100")
        print(f"Disentanglement Accuracy : {dis:6.2f}%")
        print(f"Reference Resolution     : {ref:6.2f}%")
        print("-" * 65)

        cf = results.get("counterfactual_consistency", {})
        if cf.get("total_pairs", 0) > 0:
            print(f"Counterfactual Consist.  : {cf.get('consistency_score', 0.0):6.2f}% ({cf.get('consistent_pairs')}/{cf.get('total_pairs')} pairs)")

        slot_acc = results.get("slot_accuracies", {})
        slot_f1 = results.get("slot_macro_f1", {})
        if slot_acc:
            print("\nSlot-Level Performance:")
            print(f"  {'Slot':<12} | {'Accuracy':<10} | {'Macro Precision':<16} | {'Macro Recall':<14} | {'Macro F1':<10}")
            print("  " + "-" * 70)
            for slot in ["agent", "action", "object", "tense", "polarity"]:
                acc = slot_acc.get(slot, 0.0)
                f1_data = slot_f1.get(slot, {})
                mp = f1_data.get("macro_precision", 0.0)
                mr = f1_data.get("macro_recall", 0.0)
                mf1 = f1_data.get("macro_f1", 0.0)
                print(f"  {slot:<12} | {acc:8.2f}% | {mp:14.2f}% | {mr:12.2f}% | {mf1:8.2f}%")

        prog = results.get("progressive_levels", {})
        if prog:
            print("\nProgressive Difficulty Breakdown:")
            for lvl_name, lvl_data in sorted(prog.items()):
                print(f"  {lvl_name:<10}: {lvl_data.get('accuracy', 0.0):6.2f}% (N={lvl_data.get('total')})")

        phen = results.get("phenomenon_breakdown", {})
        if phen:
            print("\nDialogue Phenomenon Breakdown:")
            for p_name, p_data in sorted(phen.items()):
                print(f"  {p_name:<14}: {p_data.get('accuracy', 0.0):6.2f}% (N={p_data.get('total')})")

        print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Project SHIFT Standard Evaluator (IOAI 2027)")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions JSON/JSONL file")
    parser.add_argument("--dataset", type=str, default=None, help="Path to dataset JSONL (with query_label or for episode metadata)")
    parser.add_argument("--labels", type=str, default=None, help="Path to ground truth labels JSON file")
    parser.add_argument("--output", type=str, default=None, help="Optional output path for detailed results JSON")
    parser.add_argument("--strict", action="store_true", help="Fail immediately on any schema validation error")
    parser.add_argument("--quiet", action="store_true", help="Suppress console summary output")

    args = parser.parse_args()

    results = Evaluator.evaluate_files(
        predictions_path=args.predictions,
        dataset_path=args.dataset,
        labels_path=args.labels,
        output_path=args.output,
        strict_schema=args.strict
    )

    if not args.quiet:
        Evaluator.print_summary(results)


if __name__ == "__main__":
    main()
