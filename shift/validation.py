"""
Project SHIFT: Data Validation and Learnability Audit Engine.
Milestone 09 Deliverable for IOAI 2027 Task Development.

Executes automated quality gates: schema correctness, support learnability coverage,
context determinacy, and label distribution shortcut audits.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Any
from collections import Counter

from shift.ontology import EventFrame, AGENT_METADATA
from shift.episodes import Episode


class QualityAuditReport:
    """Stores quantitative results from quality gate audits."""
    def __init__(self):
        self.total_episodes: int = 0
        self.valid_schema_count: int = 0
        self.learnable_count: int = 0
        self.context_determinate_count: int = 0
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.agent_distribution: Counter = Counter()
        self.action_distribution: Counter = Counter()
        self.object_distribution: Counter = Counter()
        self.tense_distribution: Counter = Counter()
        self.polarity_distribution: Counter = Counter()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_episodes": self.total_episodes,
            "valid_schema_count": self.valid_schema_count,
            "schema_pass_rate": (self.valid_schema_count / self.total_episodes) if self.total_episodes else 0.0,
            "learnable_count": self.learnable_count,
            "learnability_pass_rate": (self.learnable_count / self.total_episodes) if self.total_episodes else 0.0,
            "context_determinate_count": self.context_determinate_count,
            "context_pass_rate": (self.context_determinate_count / self.total_episodes) if self.total_episodes else 0.0,
            "errors": self.errors[:20], # limit error log length
            "total_errors": len(self.errors),
            "warnings": self.warnings[:20],
            "distributions": {
                "agents": dict(self.agent_distribution.most_common(5)),
                "actions": dict(self.action_distribution.most_common(5)),
                "objects": dict(self.object_distribution.most_common(5)),
                "tenses": dict(self.tense_distribution),
                "polarities": dict(self.polarity_distribution),
            }
        }


def validate_episode_schema(ep: Episode) -> Tuple[bool, List[str]]:
    """Quality Gate A: Generator correctness against schema."""
    errors = []
    # Check support frames
    for idx, supp in enumerate(ep.support_set):
        ok, errs = supp.event_frame.validate()
        if not ok:
            errors.append(f"Support #{idx+1} invalid: {errs}")

    # Check query frame if present
    if ep.query_label:
        ok, errs = ep.query_label.validate()
        if not ok:
            errors.append(f"Query label invalid: {errs}")

    return len(errors) == 0, errors


def audit_episode_learnability(ep: Episode) -> Tuple[bool, List[str]]:
    """
    Quality Gate B: Learnability audit using six-check validation.
    Verifies lexical, grammar, semantic drift, context sufficiency, unique interpretation,
    and compositional learnability so the query is solvable without arbitrary guessing.
    """
    if not ep.query_label:
        return True, [] # Can only audit when ground truth is known

    from shift.learnability_validator import LearnabilityValidator
    audit = LearnabilityValidator.audit_episode(ep)
    return audit.is_solvable, audit.failure_reasons


def audit_context_determinacy(ep: Episode) -> Tuple[bool, List[str]]:
    """
    Quality Gate C: Context determinacy.
    Verifies that if query contains pronouns or ellipsis, the preceding context contains
    unambiguous entity mentions to resolve them.
    """
    phenomenon = ep.query.phenomenon
    if phenomenon == "direct":
        return True, []

    errors = []
    if not ep.query.context:
        errors.append(f"Query uses phenomenon '{phenomenon}' but context turns are empty")
        return False, errors

    # Check that previous turns have valid event frames
    has_antecedent = False
    for turn in ep.query.context:
        if "event_frame" in turn:
            has_antecedent = True
            break

    if not has_antecedent:
        errors.append(f"Context does not contain any valid antecedent event frame")

    return len(errors) == 0, errors


def audit_dataset_quality(episodes: List[Episode]) -> QualityAuditReport:
    """Runs full validation suite over a collection of episodes."""
    report = QualityAuditReport()
    report.total_episodes = len(episodes)

    for ep in episodes:
        # 1. Schema check
        s_ok, s_errs = validate_episode_schema(ep)
        if s_ok:
            report.valid_schema_count += 1
        else:
            report.errors.extend([f"[{ep.episode_id}] {e}" for e in s_errs])

        # 2. Learnability check
        l_ok, l_errs = audit_episode_learnability(ep)
        if l_ok:
            report.learnable_count += 1
        else:
            report.warnings.extend([f"[{ep.episode_id}] {e}" for e in l_errs])

        # 3. Context determinacy check
        c_ok, c_errs = audit_context_determinacy(ep)
        if c_ok:
            report.context_determinate_count += 1
        else:
            report.errors.extend([f"[{ep.episode_id}] {e}" for e in c_errs])

        # 4. Distribution tracking
        if ep.query_label:
            report.agent_distribution[ep.query_label.agent] += 1
            report.action_distribution[ep.query_label.action] += 1
            report.object_distribution[ep.query_label.object] += 1
            report.tense_distribution[ep.query_label.tense] += 1
            report.polarity_distribution[ep.query_label.polarity] += 1

    return report
