"""
Project SHIFT: Semantic Adaptation Across Communities and Historical Time.
IOAI 2027 Scientific Benchmark Infrastructure.
"""

from shift.ontology import (
    EventFrame, AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity, DriftSource
)
from shift.episodes import Episode, SupportExample, QueryInstance
from shift.serialize import load_episodes_jsonl, export_episodes_jsonl, load_ground_truth_labels
from shift.schema import SchemaValidator, SchemaValidationError, get_event_frame_schema
from shift.metrics import compute_all_metrics, compute_slot_f1
from shift.learnability_validator import LearnabilityValidator, LearnabilityAuditResult
from shift.shortcut_audits import ShortcutAuditor
from shift.evaluator import Evaluator
from shift.diagnostics import ModelDiagnostics, FailureMode, DiagnosticReport
from shift.evaluation_report import ReportGenerator

__version__ = "0.3.0"
