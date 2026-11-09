"""
Project SHIFT: Formal Semantic Ontology and Event Schema.
Milestone 02 Deliverable for IOAI 2027 Task Development.

Defines the controlled semantic world: agents, actions, objects, tense,
polarity, reference types, and the strict EventFrame schema with validation.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Set, Optional, Any, Tuple


class AgentRole(str, Enum):
    SCOUT = "scout"
    ARTISAN = "artisan"
    SCHOLAR = "scholar"
    GUARDIAN = "guardian"


class AgentEntity(str, Enum):
    SCOUT_ALPHA = "scout_alpha"
    SCOUT_BETA = "scout_beta"
    ARTISAN_GAMMA = "artisan_gamma"
    ARTISAN_DELTA = "artisan_delta"
    SCHOLAR_OMEGA = "scholar_omega"
    SCHOLAR_KAPPA = "scholar_kappa"
    GUARDIAN_THETA = "guardian_theta"
    GUARDIAN_SIGMA = "guardian_sigma"
    SPEAKER = "speaker"
    ADDRESSEE = "addressee"


class ActionPredicate(str, Enum):
    INSPECT = "inspect"
    HARVEST = "harvest"
    FORGE = "forge"
    SECURE = "secure"
    RELOCATE = "relocate"
    REPAIR = "repair"
    ACTIVATE = "activate"
    ABANDON = "abandon"


class ObjectEntity(str, Enum):
    CRYSTAL_CORE = "crystal_core"
    ANCIENT_RELIC = "ancient_relic"
    SIGNAL_BEACON = "signal_beacon"
    WATER_PURIFIER = "water_purifier"
    STONE_MONOLITH = "stone_monolith"
    ENERGY_MATRIX = "energy_matrix"
    DATA_CAPSULE = "data_capsule"
    SHELTER_DOME = "shelter_dome"


class Tense(str, Enum):
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"


class Polarity(str, Enum):
    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"


class DriftSource(str, Enum):
    STABLE = "stable"
    COMMUNITY_SHIFT = "community_shift"
    HISTORICAL_SHIFT = "historical_shift"
    INTERACTION_SHIFT = "interaction_shift"


class ReferenceType(str, Enum):
    DIRECT_NAME = "direct_name"
    PRONOUN = "pronoun"
    DEFINITE_NP = "definite_np"
    DEMONSTRATIVE = "demonstrative"
    ELLIPSIS = "ellipsis"


# Agent metadata mapping role, gender/pronoun affinity, and default domains
AGENT_METADATA: Dict[str, Dict[str, Any]] = {
    AgentEntity.SCOUT_ALPHA.value: {"role": AgentRole.SCOUT.value, "gender": "masculine", "title": "Scout Alpha"},
    AgentEntity.SCOUT_BETA.value: {"role": AgentRole.SCOUT.value, "gender": "feminine", "title": "Scout Beta"},
    AgentEntity.ARTISAN_GAMMA.value: {"role": AgentRole.ARTISAN.value, "gender": "masculine", "title": "Artisan Gamma"},
    AgentEntity.ARTISAN_DELTA.value: {"role": AgentRole.ARTISAN.value, "gender": "feminine", "title": "Artisan Delta"},
    AgentEntity.SCHOLAR_OMEGA.value: {"role": AgentRole.SCHOLAR.value, "gender": "masculine", "title": "Scholar Omega"},
    AgentEntity.SCHOLAR_KAPPA.value: {"role": AgentRole.SCHOLAR.value, "gender": "feminine", "title": "Scholar Kappa"},
    AgentEntity.GUARDIAN_THETA.value: {"role": AgentRole.GUARDIAN.value, "gender": "masculine", "title": "Guardian Theta"},
    AgentEntity.GUARDIAN_SIGMA.value: {"role": AgentRole.GUARDIAN.value, "gender": "feminine", "title": "Guardian Sigma"},
    AgentEntity.SPEAKER.value: {"role": "participant", "gender": "animate", "title": "Speaker"},
    AgentEntity.ADDRESSEE.value: {"role": "participant", "gender": "animate", "title": "Addressee"},
}

OBJECT_METADATA: Dict[str, Dict[str, Any]] = {
    ObjectEntity.CRYSTAL_CORE.value: {"category": "energy", "portable": True},
    ObjectEntity.ANCIENT_RELIC.value: {"category": "artifact", "portable": True},
    ObjectEntity.SIGNAL_BEACON.value: {"category": "comms", "portable": False},
    ObjectEntity.WATER_PURIFIER.value: {"category": "utility", "portable": False},
    ObjectEntity.STONE_MONOLITH.value: {"category": "monument", "portable": False},
    ObjectEntity.ENERGY_MATRIX.value: {"category": "infrastructure", "portable": False},
    ObjectEntity.DATA_CAPSULE.value: {"category": "knowledge", "portable": True},
    ObjectEntity.SHELTER_DOME.value: {"category": "structure", "portable": False},
}

# Selectional Restrictions: Canonical preferences (used to evaluate compositional OOD splits)
CANONICAL_PREFERENCES: Dict[str, Set[str]] = {
    ActionPredicate.INSPECT.value: {o.value for o in ObjectEntity},
    ActionPredicate.HARVEST.value: {ObjectEntity.CRYSTAL_CORE.value, ObjectEntity.WATER_PURIFIER.value, ObjectEntity.ANCIENT_RELIC.value},
    ActionPredicate.FORGE.value: {ObjectEntity.SIGNAL_BEACON.value, ObjectEntity.WATER_PURIFIER.value, ObjectEntity.ENERGY_MATRIX.value, ObjectEntity.CRYSTAL_CORE.value},
    ActionPredicate.SECURE.value: {ObjectEntity.SHELTER_DOME.value, ObjectEntity.STONE_MONOLITH.value, ObjectEntity.SIGNAL_BEACON.value, ObjectEntity.DATA_CAPSULE.value},
    ActionPredicate.RELOCATE.value: {ObjectEntity.CRYSTAL_CORE.value, ObjectEntity.ANCIENT_RELIC.value, ObjectEntity.DATA_CAPSULE.value, ObjectEntity.SIGNAL_BEACON.value},
    ActionPredicate.REPAIR.value: {ObjectEntity.SIGNAL_BEACON.value, ObjectEntity.WATER_PURIFIER.value, ObjectEntity.ENERGY_MATRIX.value, ObjectEntity.SHELTER_DOME.value},
    ActionPredicate.ACTIVATE.value: {ObjectEntity.SIGNAL_BEACON.value, ObjectEntity.ENERGY_MATRIX.value, ObjectEntity.CRYSTAL_CORE.value, ObjectEntity.WATER_PURIFIER.value},
    ActionPredicate.ABANDON.value: {ObjectEntity.SHELTER_DOME.value, ObjectEntity.STONE_MONOLITH.value, ObjectEntity.ANCIENT_RELIC.value, ObjectEntity.WATER_PURIFIER.value},
}


@dataclass
class EventFrame:
    """
    Structured representation of an extracted conversational event.
    Standardized target output for Project SHIFT queries.
    """
    agent: str
    action: str
    object: str
    tense: str
    polarity: str
    resolved_references: Dict[str, str] = field(default_factory=dict)
    drift_attribution: Dict[str, str] = field(default_factory=dict)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate event frame against ontology boundaries."""
        errors: List[str] = []

        valid_agents = {a.value for a in AgentEntity}
        if self.agent not in valid_agents:
            errors.append(f"Invalid agent '{self.agent}'. Expected one of: {sorted(valid_agents)}")

        valid_actions = {a.value for a in ActionPredicate}
        if self.action not in valid_actions:
            errors.append(f"Invalid action '{self.action}'. Expected one of: {sorted(valid_actions)}")

        valid_objects = {o.value for o in ObjectEntity}
        if self.object not in valid_objects:
            errors.append(f"Invalid object '{self.object}'. Expected one of: {sorted(valid_objects)}")

        valid_tenses = {t.value for t in Tense}
        if self.tense not in valid_tenses:
            errors.append(f"Invalid tense '{self.tense}'. Expected one of: {sorted(valid_tenses)}")

        valid_polarities = {p.value for p in Polarity}
        if self.polarity not in valid_polarities:
            errors.append(f"Invalid polarity '{self.polarity}'. Expected one of: {sorted(valid_polarities)}")

        if not isinstance(self.resolved_references, dict):
            errors.append("resolved_references must be a dictionary")
        else:
            for ref_key, target in self.resolved_references.items():
                if not isinstance(ref_key, str) or not isinstance(target, str):
                    errors.append(f"Invalid reference pair ({ref_key}: {target}). Both must be strings.")

        if not isinstance(self.drift_attribution, dict):
            errors.append("drift_attribution must be a dictionary")
        else:
            valid_sources = {s.value for s in DriftSource}
            for slot_key, source in self.drift_attribution.items():
                if source not in valid_sources:
                    errors.append(f"Invalid drift attribution '{source}' for slot '{slot_key}'. Expected one of: {sorted(valid_sources)}")

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary matching competition format."""
        return {
            "agent": self.agent,
            "action": self.action,
            "object": self.object,
            "tense": self.tense,
            "polarity": self.polarity,
            "resolved_references": dict(self.resolved_references),
            "drift_attribution": dict(self.drift_attribution)
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EventFrame:
        """Instantiate EventFrame from dictionary with validation."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")
        frame = cls(
            agent=str(data.get("agent", "")),
            action=str(data.get("action", "")),
            object=str(data.get("object", "")),
            tense=str(data.get("tense", "")),
            polarity=str(data.get("polarity", "")),
            resolved_references=dict(data.get("resolved_references", {})),
            drift_attribution=dict(data.get("drift_attribution", {}))
        )
        is_valid, errors = frame.validate()
        if not is_valid:
            raise ValueError(f"EventFrame schema validation failed: {'; '.join(errors)}")
        return frame

    def canonical_tuple(self) -> Tuple[str, str, str, str, str, Tuple[Tuple[str, str], ...]]:
        """Hashable canonical representation for exact match comparison."""
        sorted_refs = tuple(sorted(self.resolved_references.items()))
        return (self.agent, self.action, self.object, self.tense, self.polarity, sorted_refs)

    def exact_match(self, other: EventFrame) -> bool:
        """Checks if all slots and resolved references match exactly."""
        if not isinstance(other, EventFrame):
            return False
        return self.canonical_tuple() == other.canonical_tuple()

    def slot_match_dict(self, other: EventFrame) -> Dict[str, bool]:
        """Returns per-slot boolean matches."""
        if not isinstance(other, EventFrame):
            return {
                "agent": False, "action": False, "object": False,
                "tense": False, "polarity": False, "resolved_references": False,
                "drift_attribution": False
            }
        return {
            "agent": self.agent == other.agent,
            "action": self.action == other.action,
            "object": self.object == other.object,
            "tense": self.tense == other.tense,
            "polarity": self.polarity == other.polarity,
            "resolved_references": self.resolved_references == other.resolved_references,
            "drift_attribution": self.drift_attribution == other.drift_attribution
        }


def get_json_schema() -> Dict[str, Any]:
    """Returns official JSON Schema for EventFrame evaluation and validation."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SHIFT_EventFrame",
        "type": "object",
        "required": ["agent", "action", "object", "tense", "polarity", "resolved_references"],
        "properties": {
            "agent": {
                "type": "string",
                "enum": [a.value for a in AgentEntity]
            },
            "action": {
                "type": "string",
                "enum": [a.value for a in ActionPredicate]
            },
            "object": {
                "type": "string",
                "enum": [o.value for o in ObjectEntity]
            },
            "tense": {
                "type": "string",
                "enum": [t.value for t in Tense]
            },
            "polarity": {
                "type": "string",
                "enum": [p.value for p in Polarity]
            },
            "resolved_references": {
                "type": "object",
                "additionalProperties": {
                    "type": "string"
                }
            }
        },
        "additionalProperties": False
    }
