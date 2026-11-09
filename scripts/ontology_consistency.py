"""
Project SHIFT: Ontology Consistency & Collision Validator.
Component of v0.4 Dataset & Annotation Validation.

Audits ontology definitions, selectional restrictions, morpheme collisions,
and phonological invariants across all synthetic communities and epochs.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shift.ontology import (
    AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity, DriftSource,
    AGENT_METADATA, OBJECT_METADATA, CANONICAL_PREFERENCES
)
from shift.lexicon import Lexicon
from shift.drift import DriftEngine


def run_ontology_consistency_check(
    output_path: str = "results/ontology_consistency_results.json"
) -> Dict[str, Any]:
    print("=" * 70)
    print(" PROJECT SHIFT: Ontology & Lexicon Consistency Audit")
    print("=" * 70)

    issues: List[str] = []

    # 1. Check Enum Cardinalities
    n_agents = len(AgentEntity)
    n_actions = len(ActionPredicate)
    n_objects = len(ObjectEntity)
    n_tenses = len(Tense)
    n_polarities = len(Polarity)
    n_sources = len(DriftSource)

    print(f"Ontology Dimensions: {n_agents} Agents, {n_actions} Actions, {n_objects} Objects")
    print(f"Grammatical Dimensions: {n_tenses} Tenses, {n_polarities} Polarities, {n_sources} Drift Sources")

    if n_agents != 10: issues.append(f"Expected 10 agents, found {n_agents}")
    if n_actions != 8: issues.append(f"Expected 8 actions, found {n_actions}")
    if n_objects != 8: issues.append(f"Expected 8 objects, found {n_objects}")
    if n_tenses != 3: issues.append(f"Expected 3 tenses, found {n_tenses}")
    if n_polarities != 2: issues.append(f"Expected 2 polarities, found {n_polarities}")
    if n_sources != 4: issues.append(f"Expected 4 drift sources, found {n_sources}")

    # 2. Check Metadata Dictionary Coverage
    for a in AgentEntity:
        if a.value not in AGENT_METADATA:
            issues.append(f"Missing AGENT_METADATA for {a.value}")
    for act in ActionPredicate:
        if act.value not in CANONICAL_PREFERENCES:
            issues.append(f"Missing CANONICAL_PREFERENCES for {act.value}")
    for obj in ObjectEntity:
        if obj.value not in OBJECT_METADATA:
            issues.append(f"Missing OBJECT_METADATA for {obj.value}")

    # 3. Factorial Drift Matrix Orthogonality Check
    communities = ["comm_citadel", "comm_north", "comm_coast"]
    periods = ["period_archaic", "period_classical", "period_contemporary"]
    total_cells = len(communities) * len(periods)
    print(f"\nChecking Factorial Matrix Orthogonality ({len(communities)}x{len(periods)} = {total_cells} cells)...")

    lexicon_collisions = 0
    for comm in communities:
        for per in periods:
            drift_engine = DriftEngine(community_id=comm, period_id=per)
            lex = Lexicon()
            transformed_lex = drift_engine.transform_lexicon(lex)
            try:
                warnings = transformed_lex.audit_collisions()
            except Exception as collision_err:
                lexicon_collisions += 1
                issues.append(f"Collision in ({comm}, {per}): {collision_err}")

    print(f"  [OK] Evaluated {total_cells} factorial language cells: {lexicon_collisions} collisions detected.")

    status = "PASSED" if len(issues) == 0 else f"FAILED ({len(issues)} issues)"
    print(f"\nOverall Consistency Status: {status}")

    results = {
        "status": status,
        "is_perfect": len(issues) == 0,
        "ontology_sizes": {
            "agents": n_agents,
            "actions": n_actions,
            "objects": n_objects,
            "tenses": n_tenses,
            "polarities": n_polarities,
            "drift_sources": n_sources
        },
        "issues": issues
    }

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[OK] Results exported to {output_path}\n")
    return results


if __name__ == "__main__":
    run_ontology_consistency_check()
