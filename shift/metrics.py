"""
Project SHIFT: Objective Scoring Metrics Engine.
Component 6 of SHIFT v0.3 Scientific Benchmark Infrastructure.

Computes exact match, slot-level macro/micro F1, disentanglement accuracy,
coreference accuracy, counterfactual consistency, and progressive competition levels.
"""

from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple, Any
from collections import defaultdict, Counter

from shift.ontology import EventFrame
from shift.episodes import Episode


def compute_slot_f1(
    predictions: Dict[str, EventFrame],
    ground_truth: Dict[str, EventFrame],
    slot_name: str
) -> Dict[str, float]:
    """Computes precision, recall, and macro F1 for a specific event frame slot."""
    labels: Set[str] = set()
    for gt in ground_truth.values():
        val = getattr(gt, slot_name, None)
        if val: labels.add(val)

    class_precisions = []
    class_recalls = []
    class_f1s = []

    for label in sorted(labels):
        tp = 0
        fp = 0
        fn = 0

        for ep_id, true_frame in ground_truth.items():
            true_val = getattr(true_frame, slot_name, None)
            pred_frame = predictions.get(ep_id)
            pred_val = getattr(pred_frame, slot_name, None) if pred_frame else None

            if true_val == label and pred_val == label:
                tp += 1
            elif true_val != label and pred_val == label:
                fp += 1
            elif true_val == label and pred_val != label:
                fn += 1

        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        class_precisions.append(prec)
        class_recalls.append(rec)
        class_f1s.append(f1)

    macro_f1 = (sum(class_f1s) / len(class_f1s)) if class_f1s else 0.0
    macro_prec = (sum(class_precisions) / len(class_precisions)) if class_precisions else 0.0
    macro_rec = (sum(class_recalls) / len(class_recalls)) if class_recalls else 0.0

    return {
        "macro_precision": round(macro_prec * 100.0, 2),
        "macro_recall": round(macro_rec * 100.0, 2),
        "macro_f1": round(macro_f1 * 100.0, 2)
    }


def compute_all_metrics(
    predictions: Dict[str, EventFrame],
    ground_truth: Dict[str, EventFrame],
    episodes_by_id: Optional[Dict[str, Episode]] = None
) -> Dict[str, Any]:
    """
    Computes full comprehensive evaluation metrics for Project SHIFT v0.3.
    """
    total = len(ground_truth)
    if total == 0:
        return {"error": "Ground truth is empty"}

    exact_matches = 0
    ref_matches = 0
    disentangle_matches = 0
    slot_matches = {"agent": 0, "action": 0, "object": 0, "tense": 0, "polarity": 0}

    level_stats = defaultdict(lambda: {"total": 0, "exact": 0})
    phenomenon_stats = defaultdict(lambda: {"total": 0, "exact": 0})
    cf_pairs = defaultdict(dict)

    for ep_id, true_frame in ground_truth.items():
        pred_frame = predictions.get(ep_id)
        if pred_frame is None:
            continue

        is_exact = true_frame.exact_match(pred_frame)
        if is_exact:
            exact_matches += 1

        for slot in ["agent", "action", "object", "tense", "polarity"]:
            if getattr(true_frame, slot) == getattr(pred_frame, slot):
                slot_matches[slot] += 1

        if true_frame.resolved_references == pred_frame.resolved_references:
            ref_matches += 1

        if true_frame.drift_attribution and pred_frame.drift_attribution:
            if true_frame.drift_attribution == pred_frame.drift_attribution:
                disentangle_matches += 1

        if episodes_by_id and ep_id in episodes_by_id:
            ep = episodes_by_id[ep_id]
            # Phenomenon tracking
            p = ep.query.phenomenon
            phenomenon_stats[p]["total"] += 1
            if is_exact: phenomenon_stats[p]["exact"] += 1

            # Progressive Level tracking
            lvl = ep.difficulty_level
            level_stats[lvl]["total"] += 1
            if is_exact: level_stats[lvl]["exact"] += 1

            # Counterfactual Pair tracking
            if ep.counterfactual_pair_id:
                branch = ep.metadata.get("counterfactual_branch", "A" if ep_id.endswith("_A") else "B")
                cf_pairs[ep.counterfactual_pair_id][branch] = {
                    "is_exact": is_exact,
                    "pred": pred_frame,
                    "gt": true_frame
                }

    # Slot F1 Scores
    slot_f1_scores = {}
    for slot in ["agent", "action", "object", "tense", "polarity"]:
        slot_f1_scores[slot] = compute_slot_f1(predictions, ground_truth, slot)

    # Redesigned Counterfactual Correctness & Consistency Metrics
    total_cf = len(cf_pairs)
    branch_a_correct_cnt = 0
    branch_b_correct_cnt = 0
    joint_correct_cnt = 0
    sensitive_cnt = 0

    for pair_id, branches in cf_pairs.items():
        if "A" in branches and "B" in branches:
            b_a = branches["A"]
            b_b = branches["B"]
            if b_a["is_exact"]: branch_a_correct_cnt += 1
            if b_b["is_exact"]: branch_b_correct_cnt += 1
            if b_a["is_exact"] and b_b["is_exact"]:
                joint_correct_cnt += 1

            # Perturbation sensitivity: if ground truths differ, predictions must differ
            gt_differ = not b_a["gt"].exact_match(b_b["gt"])
            pred_differ = not b_a["pred"].exact_match(b_b["pred"]) if (b_a["pred"] and b_b["pred"]) else False
            if (gt_differ and pred_differ) or (not gt_differ and not pred_differ):
                sensitive_cnt += 1

    branch_a_acc = round(branch_a_correct_cnt / total_cf * 100.0, 2) if total_cf else 100.0
    branch_b_acc = round(branch_b_correct_cnt / total_cf * 100.0, 2) if total_cf else 100.0
    joint_acc = round(joint_correct_cnt / total_cf * 100.0, 2) if total_cf else 100.0
    sensitivity_acc = round(sensitive_cnt / total_cf * 100.0, 2) if total_cf else 100.0

    # Consistency score combines joint accuracy and sensitivity
    cf_score = round((joint_acc * 0.7 + sensitivity_acc * 0.3), 2) if total_cf else 100.0

    # Progressive Level Scores
    level_scores = {}
    for lvl in sorted(level_stats.keys()):
        st = level_stats[lvl]
        acc = round(st["exact"] / st["total"] * 100.0, 2) if st["total"] else 0.0
        level_scores[f"Level_{lvl}"] = {"total": st["total"], "accuracy": acc}

    # Competition Composite Score (Partial Credit Formula)
    mean_slot_f1 = sum(v["macro_f1"] for v in slot_f1_scores.values()) / len(slot_f1_scores)
    em_acc = round(exact_matches / total * 100.0, 2)
    dis_acc = round(disentangle_matches / total * 100.0, 2)
    ref_acc = round(ref_matches / total * 100.0, 2)

    composite_score = round(
        0.40 * em_acc +
        0.25 * mean_slot_f1 +
        0.15 * dis_acc +
        0.10 * ref_acc +
        0.10 * cf_score,
        2
    )

    return {
        "total_episodes": total,
        "exact_match_accuracy": em_acc,
        "macro_slot_f1": round(mean_slot_f1, 2),
        "composite_competition_score": composite_score,
        "disentanglement_accuracy": dis_acc,
        "reference_resolution_accuracy": ref_acc,
        "slot_accuracies": {k: round(v / total * 100.0, 2) for k, v in slot_matches.items()},
        "slot_macro_f1": slot_f1_scores,
        "counterfactual_consistency": {
            "total_pairs": total_cf,
            "branch_a_accuracy": branch_a_acc,
            "branch_b_accuracy": branch_b_acc,
            "joint_accuracy": joint_acc,
            "perturbation_sensitivity": sensitivity_acc,
            "consistency_score": cf_score
        },
        "progressive_levels": level_scores,
        "phenomenon_breakdown": {
            k: {
                "total": v["total"],
                "accuracy": round(v["exact"] / v["total"] * 100.0, 2) if v["total"] else 0.0
            }
            for k, v in phenomenon_stats.items()
        }
    }
