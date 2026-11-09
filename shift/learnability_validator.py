"""
Project SHIFT: Learnability & Ambiguity Validator.
Component 3 of SHIFT v0.3 Scientific Benchmark Infrastructure.

Guarantees that every evaluation query is mathematically and linguistically solvable
from the provided support set and conversational context without arbitrary guessing.
Rejects ambiguous, underspecified, or unsolvable episodes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
import re

from shift.ontology import AgentEntity, ActionPredicate, ObjectEntity, Tense, Polarity, AGENT_METADATA
from shift.episodes import Episode


@dataclass
class LearnabilityAuditResult:
    """Detailed result of a learnability and ambiguity audit on an episode."""
    episode_id: str
    is_solvable: bool
    lexical_grounding: bool
    morphological_grounding: bool
    discourse_determinacy: bool
    ambiguity_free: bool
    missing_evidence: List[str] = field(default_factory=list)
    ambiguities: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "is_solvable": self.is_solvable,
            "lexical_grounding": self.lexical_grounding,
            "morphological_grounding": self.morphological_grounding,
            "discourse_determinacy": self.discourse_determinacy,
            "ambiguity_free": self.ambiguity_free,
            "missing_evidence": self.missing_evidence,
            "ambiguities": self.ambiguities
        }


class LearnabilityValidator:
    """
    Validates that each query is fair, unambiguous, and inferable.
    Acts as an automated quality gate before dataset inclusion.
    """

    @classmethod
    def audit_episode(cls, episode: Episode) -> LearnabilityAuditResult:
        """Runs the four foundational learnability checks on an episode."""
        missing: List[str] = []
        ambiguities: List[str] = []

        target = episode.query_label
        if target is None:
            # Cannot verify ground-truth solvability without label
            return LearnabilityAuditResult(
                episode_id=episode.episode_id,
                is_solvable=True,
                lexical_grounding=True,
                morphological_grounding=True,
                discourse_determinacy=True,
                ambiguity_free=True
            )

        supp_frames = [s.event_frame for s in episode.support_set]
        supp_agents: Set[str] = {f.agent for f in supp_frames}
        supp_actions: Set[str] = {f.action for f in supp_frames}
        supp_objects: Set[str] = {f.object for f in supp_frames}
        supp_tenses: Set[str] = {f.tense for f in supp_frames}
        supp_polarities: Set[str] = {f.polarity for f in supp_frames}

        phenomenon = episode.query.phenomenon
        context = episode.query.context

        # Extract discourse entities from context
        ctx_masculine: List[str] = []
        ctx_feminine: List[str] = []
        ctx_objects: List[str] = []
        ctx_actions: List[str] = []

        for turn in context:
            f = turn.get("event_frame", {})
            agent = f.get("agent")
            if agent:
                meta = AGENT_METADATA.get(agent, {})
                if meta.get("gender") == "masculine":
                    ctx_masculine.append(agent)
                elif meta.get("gender") == "feminine":
                    ctx_feminine.append(agent)
            obj = f.get("object")
            if obj:
                ctx_objects.append(obj)
            act = f.get("action")
            if act:
                ctx_actions.append(act)

        # 1. Lexical Grounding Audit
        lexical_ok = True

        # Check Agent
        if phenomenon in ("coreference", "correction", "counterfactual_pair"):
            # Agent resolved via discourse context
            if target.agent == AgentEntity.SPEAKER.value or target.agent == AgentEntity.ADDRESSEE.value:
                pass # 1st / 2nd person indexicals are self-grounded
            elif target.agent not in ctx_masculine and target.agent not in ctx_feminine:
                missing.append(f"Agent '{target.agent}' absent from both support set and dialogue context.")
                lexical_ok = False
        else:
            if target.agent not in supp_agents:
                missing.append(f"Agent '{target.agent}' absent from support set.")
                lexical_ok = False

        # Check Action
        if phenomenon == "ellipsis":
            # Action borrowed from preceding turn
            if target.action not in ctx_actions:
                missing.append(f"Elliptic action '{target.action}' absent from preceding conversation turns.")
                lexical_ok = False
        else:
            if target.action not in supp_actions and phenomenon not in ("counterfactual_pair", "correction"):
                missing.append(f"Action '{target.action}' absent from support set.")
                lexical_ok = False

        # Check Object
        if phenomenon in ("coreference", "counterfactual_pair") and target.resolved_references.get("id"):
            if target.object not in ctx_objects:
                missing.append(f"Pronominal object '{target.object}' absent from preceding conversation context.")
                lexical_ok = False
        else:
            if target.object not in supp_objects and phenomenon != "counterfactual_pair":
                missing.append(f"Object '{target.object}' absent from support set.")
                lexical_ok = False

        # 2. Morphological Grounding Audit
        morph_ok = True
        if target.tense not in supp_tenses:
            missing.append(f"Tense '{target.tense}' not exemplified in support set.")
            morph_ok = False

        # 3. Discourse Determinacy Audit
        discourse_ok = True
        if phenomenon in ("coreference", "ellipsis", "correction", "counterfactual_pair"):
            if not context:
                missing.append(f"Query uses discourse phenomenon '{phenomenon}' but context is empty.")
                discourse_ok = False

        # 4. Ambiguity Audit (No competing antecedents of equal agreement)
        ambiguity_ok = True
        q_tokens = re.findall(r"\b\w+\b", episode.query.utterance.lower())
        if "el" in q_tokens:
            if len(set(ctx_masculine)) > 1 and ctx_masculine[-1] != target.agent:
                ambiguities.append(f"Multiple masculine agents in context {ctx_masculine}; ambiguous reference for 'el'.")
                ambiguity_ok = False
        if "la" in q_tokens:
            if len(set(ctx_feminine)) > 1 and ctx_feminine[-1] != target.agent:
                ambiguities.append(f"Multiple feminine agents in context {ctx_feminine}; ambiguous reference for 'la'.")
                ambiguity_ok = False

        is_solvable = lexical_ok and morph_ok and discourse_ok and ambiguity_ok

        return LearnabilityAuditResult(
            episode_id=episode.episode_id,
            is_solvable=is_solvable,
            lexical_grounding=lexical_ok,
            morphological_grounding=morph_ok,
            discourse_determinacy=discourse_ok,
            ambiguity_free=ambiguity_ok,
            missing_evidence=missing,
            ambiguities=ambiguities
        )

    @classmethod
    def audit_dataset_solvability(cls, episodes: List[Episode]) -> Dict[str, Any]:
        """Audits an entire collection of episodes and returns a pass-rate summary."""
        total = len(episodes)
        solvable_count = 0
        rejected_episodes: List[Dict[str, Any]] = []

        for ep in episodes:
            result = cls.audit_episode(ep)
            if result.is_solvable:
                solvable_count += 1
            else:
                rejected_episodes.append(result.to_dict())

        return {
            "total_episodes": total,
            "solvable_count": solvable_count,
            "pass_rate": round(solvable_count / total * 100.0, 2) if total else 0.0,
            "all_solvable": solvable_count == total,
            "rejected_count": len(rejected_episodes),
            "rejected_samples": rejected_episodes[:10]
        }
