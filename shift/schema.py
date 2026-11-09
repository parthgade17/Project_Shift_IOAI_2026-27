"""
Project SHIFT: Standard Prediction Schema & Strict Validation.
Component 5 of SHIFT v0.3 Scientific Benchmark Infrastructure.

Defines the official event representation, JSON schema, serialization,
and detailed structural validation for competition submissions and benchmark evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Optional, Tuple, Any
import json

from shift.ontology import (
    AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity, DriftSource
)


@dataclass
class SchemaValidationError:
    """Detailed error object for schema validation failures."""
    field: str
    message: str
    received_value: Any
    allowed_values: Optional[List[Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "field": self.field,
            "message": self.message,
            "received_value": self.received_value
        }
        if self.allowed_values is not None:
            d["allowed_values"] = self.allowed_values
        return d

    def __str__(self) -> str:
        if self.allowed_values:
            return f"[{self.field}] {self.message} (Got: '{self.received_value}', Allowed: {self.allowed_values})"
        return f"[{self.field}] {self.message} (Got: '{self.received_value}')"


def get_event_frame_schema() -> Dict[str, Any]:
    """Returns official Draft-07 JSON Schema for standard EventFrame prediction format."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SHIFT_EventFrame_v0.3",
        "type": "object",
        "required": ["agent", "action", "object", "tense", "polarity", "resolved_references"],
        "properties": {
            "agent": {
                "type": "string",
                "enum": [a.value for a in AgentEntity],
                "description": "Canonical agent performing the action."
            },
            "action": {
                "type": "string",
                "enum": [a.value for a in ActionPredicate],
                "description": "Canonical action predicate."
            },
            "object": {
                "type": "string",
                "enum": [o.value for o in ObjectEntity],
                "description": "Canonical object acted upon."
            },
            "tense": {
                "type": "string",
                "enum": [t.value for t in Tense],
                "description": "Event temporal anchor ('past', 'present', 'future')."
            },
            "polarity": {
                "type": "string",
                "enum": [p.value for p in Polarity],
                "description": "Affirmative or negative polarity."
            },
            "resolved_references": {
                "type": "object",
                "additionalProperties": { "type": "string" },
                "description": "Mapping from surface referring tokens/pronouns to canonical entities."
            },
            "drift_attribution": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                    "enum": [s.value for s in DriftSource]
                },
                "description": "Optional attribution of causal shift factor per slot."
            }
        },
        "additionalProperties": False
    }


class SchemaValidator:
    """Strict validator for competition submissions and dataset event frames."""

    @classmethod
    def validate_event_dict(cls, data: Any) -> Tuple[bool, List[SchemaValidationError]]:
        """Validates a raw dictionary against the EventFrame schema."""
        errors: List[SchemaValidationError] = []

        if not isinstance(data, dict):
            errors.append(SchemaValidationError(
                field="root",
                message="Expected top-level JSON object",
                received_value=type(data).__name__
            ))
            return False, errors

        # Required fields
        required = ["agent", "action", "object", "tense", "polarity", "resolved_references"]
        for req in required:
            if req not in data:
                errors.append(SchemaValidationError(
                    field=req,
                    message=f"Missing required field '{req}'",
                    received_value=None
                ))

        # Enum checks
        valid_agents = sorted([a.value for a in AgentEntity])
        if "agent" in data and data["agent"] not in valid_agents:
            errors.append(SchemaValidationError(
                field="agent",
                message=f"Invalid agent '{data['agent']}'",
                received_value=data["agent"],
                allowed_values=valid_agents
            ))

        valid_actions = sorted([a.value for a in ActionPredicate])
        if "action" in data and data["action"] not in valid_actions:
            errors.append(SchemaValidationError(
                field="action",
                message=f"Invalid action '{data['action']}'",
                received_value=data["action"],
                allowed_values=valid_actions
            ))

        valid_objects = sorted([o.value for o in ObjectEntity])
        if "object" in data and data["object"] not in valid_objects:
            errors.append(SchemaValidationError(
                field="object",
                message=f"Invalid object '{data['object']}'",
                received_value=data["object"],
                allowed_values=valid_objects
            ))

        valid_tenses = sorted([t.value for t in Tense])
        if "tense" in data and data["tense"] not in valid_tenses:
            errors.append(SchemaValidationError(
                field="tense",
                message=f"Invalid tense '{data['tense']}'",
                received_value=data["tense"],
                allowed_values=valid_tenses
            ))

        valid_polarities = sorted([p.value for p in Polarity])
        if "polarity" in data and data["polarity"] not in valid_polarities:
            errors.append(SchemaValidationError(
                field="polarity",
                message=f"Invalid polarity '{data['polarity']}'",
                received_value=data["polarity"],
                allowed_values=valid_polarities
            ))

        # Resolved references format
        if "resolved_references" in data:
            if not isinstance(data["resolved_references"], dict):
                errors.append(SchemaValidationError(
                    field="resolved_references",
                    message="resolved_references must be a dictionary",
                    received_value=type(data["resolved_references"]).__name__
                ))
            else:
                for k, v in data["resolved_references"].items():
                    if not isinstance(k, str) or not isinstance(v, str):
                        errors.append(SchemaValidationError(
                            field="resolved_references",
                            message="All keys and values in resolved_references must be strings",
                            received_value=f"({type(k).__name__}: {type(v).__name__})"
                        ))

        # Drift attribution check (optional)
        if "drift_attribution" in data and data["drift_attribution"]:
            if not isinstance(data["drift_attribution"], dict):
                errors.append(SchemaValidationError(
                    field="drift_attribution",
                    message="drift_attribution must be a dictionary",
                    received_value=type(data["drift_attribution"]).__name__
                ))
            else:
                valid_sources = sorted([s.value for s in DriftSource])
                for slot_k, src_v in data["drift_attribution"].items():
                    if src_v not in valid_sources:
                        errors.append(SchemaValidationError(
                            field=f"drift_attribution.{slot_k}",
                            message=f"Invalid drift attribution '{src_v}'",
                            received_value=src_v,
                            allowed_values=valid_sources
                        ))

        return len(errors) == 0, errors
