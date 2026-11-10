"""
Project SHIFT: Episodic Dataset Generator.
Milestone 07 Deliverable for IOAI 2027 Task Development.

Generates self-contained learning episodes featuring community/period metadata,
a small labelled support set guaranteeing learnability, dialogue context, and target queries.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from shift.ontology import (
    EventFrame, AgentEntity, ActionPredicate, ObjectEntity,
    Tense, Polarity, ReferenceType, AGENT_METADATA,
    deterministic_seed, DifficultyLevel
)
from shift.lexicon import Lexicon, generate_random_lexicon
from shift.drift import (
    DriftEngine, CommunityId, HistoricalPeriod,
    COMMUNITY_CONFIGS, HISTORICAL_CONFIGS
)
from shift.grammar import GrammarEngine
from shift.dialogue import DialogueGenerator, DialogueTurn


@dataclass
class SupportExample:
    """A labelled support example provided in an episode."""
    example_id: int
    utterance: str
    event_frame: EventFrame
    context: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_gold_context: bool = True) -> Dict[str, Any]:
        clean_ctx = []
        for t in self.context:
            if isinstance(t, dict):
                if include_gold_context:
                    clean_ctx.append(dict(t))
                else:
                    clean_ctx.append({k: v for k, v in t.items() if k not in ("event_frame", "reference_type_used")})
            elif hasattr(t, "to_dict"):
                clean_ctx.append(t.to_dict(include_gold=include_gold_context))
            else:
                clean_ctx.append(t)

        return {
            "example_id": self.example_id,
            "utterance": self.utterance,
            "event_frame": self.event_frame.to_dict(),
            "context": clean_ctx
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SupportExample:
        return cls(
            example_id=data["example_id"],
            utterance=data["utterance"],
            event_frame=EventFrame.from_dict(data["event_frame"]),
            context=data.get("context", [])
        )


@dataclass
class QueryInstance:
    """An unlabelled query for a contestant or AI system to interpret."""
    utterance: str
    context: List[Dict[str, Any]] = field(default_factory=list)
    phenomenon: str = "direct"

    def to_dict(self, include_gold: bool = True) -> Dict[str, Any]:
        """Serializes query. If include_gold=False, strips gold event_frame from context turns."""
        clean_ctx = []
        for t in self.context:
            if isinstance(t, dict):
                if include_gold:
                    clean_ctx.append(dict(t))
                else:
                    clean_ctx.append({k: v for k, v in t.items() if k not in ("event_frame", "reference_type_used")})
            elif hasattr(t, "to_dict"):
                clean_ctx.append(t.to_dict(include_gold=include_gold))
            else:
                clean_ctx.append(t)

        return {
            "utterance": self.utterance,
            "context": clean_ctx,
            "phenomenon": self.phenomenon
        }

    def to_public_dict(self) -> Dict[str, Any]:
        """Contestant-visible view with zero gold annotations in context turns."""
        return self.to_dict(include_gold=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryInstance:
        return cls(
            utterance=data["utterance"],
            context=data.get("context", []),
            phenomenon=data.get("phenomenon", "direct")
        )


@dataclass
class Episode:
    """
    A complete evaluation episode.
    Contains metadata, support set, unlabelled query, and optional ground-truth label.
    """
    episode_id: str
    community_id: str
    period_id: str
    word_order: str
    support_set: List[SupportExample]
    query: QueryInstance
    query_label: Optional[EventFrame] = None
    drift_attribution: Dict[str, str] = field(default_factory=dict)
    counterfactual_pair_id: Optional[str] = None
    difficulty_level: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_reference_dict(self) -> Dict[str, Any]:
        """Full author/evaluator serialization including ground-truth label and context frames."""
        return self.to_dict(strip_label=False, contestant_view=False)

    def to_contestant_dict(self) -> Dict[str, Any]:
        """Strictly sanitized public serialization with ZERO gold query labels and ZERO context event frames."""
        return self.to_dict(strip_label=True, contestant_view=True)

    def to_dict(self, strip_label: bool = False, contestant_view: bool = False) -> Dict[str, Any]:
        is_contestant = contestant_view or strip_label
        data: Dict[str, Any] = {
            "episode_id": self.episode_id,
            "community_id": self.community_id,
            "period_id": self.period_id,
            "word_order": self.word_order,
            "support_set": [s.to_dict(include_gold_context=not is_contestant) for s in self.support_set],
            "query": self.query.to_dict(include_gold=not is_contestant),
            "difficulty_level": self.difficulty_level,
            "metadata": dict(self.metadata)
        }
        if self.counterfactual_pair_id:
            data["counterfactual_pair_id"] = self.counterfactual_pair_id
        if not is_contestant:
            if self.query_label is not None:
                data["query_label"] = self.query_label.to_dict()
            if self.drift_attribution:
                data["drift_attribution"] = dict(self.drift_attribution)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Episode:
        q_label = None
        if "query_label" in data and data["query_label"] is not None:
            q_label = EventFrame.from_dict(data["query_label"])

        return cls(
            episode_id=data["episode_id"],
            community_id=data["community_id"],
            period_id=data["period_id"],
            word_order=data.get("word_order", "SOV"),
            support_set=[SupportExample.from_dict(s) for s in data.get("support_set", [])],
            query=QueryInstance.from_dict(data["query"]),
            query_label=q_label,
            drift_attribution=dict(data.get("drift_attribution", {})),
            counterfactual_pair_id=data.get("counterfactual_pair_id"),
            difficulty_level=int(data.get("difficulty_level", 1)),
            metadata=data.get("metadata", {})
        )


class EpisodeGenerator:
    """
    Generates well-formed episodes guaranteed to satisfy learnability criteria.
    Ensures support set provides necessary coverage of vocabulary, grammar, and discourse patterns.
    """
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.rng = random.Random(base_seed)

    def get_grammar_for_episode(
        self,
        episode_id: str,
        community_id: str = CommunityId.CITADEL.value,
        period_id: str = HistoricalPeriod.CLASSICAL.value
    ) -> GrammarEngine:
        """Derives the exact grammar engine for an episode based on base_seed."""
        ep_seed = deterministic_seed(f"{self.base_seed}_{episode_id}_{community_id}_{period_id}")
        base_lex = Lexicon(seed=ep_seed)
        drift = DriftEngine(community_id=community_id, period_id=period_id)
        drifted_lex = drift.transform_lexicon(base_lex)
        return GrammarEngine(
            lexicon=drifted_lex,
            community_profile=drift.community_profile,
            historical_profile=drift.historical_profile
        )

    def generate_single_episode(
        self,
        episode_id: str,
        community_id: str = CommunityId.CITADEL.value,
        period_id: str = HistoricalPeriod.CLASSICAL.value,
        k_support: int = 8,
        phenomenon: str = "coreference",
        forced_action: Optional[str] = None,
        forced_object: Optional[str] = None,
        forced_difficulty_level: Optional[int] = None
    ) -> Episode:
        """
        Builds a single episode with k_support labelled support examples and 1 query.
        Guarantees that query morphemes are present in the support set.
        """
        ep_seed = deterministic_seed(f"{self.base_seed}_{episode_id}_{community_id}_{period_id}")
        grammar = self.get_grammar_for_episode(episode_id, community_id, period_id)
        drift = DriftEngine(community_id=community_id, period_id=period_id)
        dialogue_gen = DialogueGenerator(grammar=grammar, seed=ep_seed)

        # 1. Generate target query and context
        context_turns, query_turn = dialogue_gen.generate_dialogue_episode(
            phenomenon=phenomenon,
            forced_action=forced_action,
            forced_object=forced_object
        )

        target_frame = query_turn.event_frame
        target_utterance = query_turn.utterance
        target_context = [t.to_dict() for t in context_turns]

        # 2. Build support set ensuring learnability coverage of target query
        support_examples: List[SupportExample] = []

        all_agents = [a.value for a in AgentEntity if a not in (AgentEntity.SPEAKER, AgentEntity.ADDRESSEE)]
        all_actions = [a.value for a in ActionPredicate]
        all_objects = [o.value for o in ObjectEntity]
        all_tenses = [t.value for t in Tense]
        all_polarities = [p.value for p in Polarity]

        # Ensure the query's agent, action, object, tense, and polarity appear in support
        critical_components = [
            (target_frame.agent, self.rng.choice(all_actions), self.rng.choice(all_objects), target_frame.tense, Polarity.AFFIRMATIVE.value),
            (self.rng.choice(all_agents), target_frame.action, self.rng.choice(all_objects), self.rng.choice(all_tenses), Polarity.AFFIRMATIVE.value),
            (self.rng.choice(all_agents), self.rng.choice(all_actions), target_frame.object, self.rng.choice(all_tenses), Polarity.AFFIRMATIVE.value),
            (self.rng.choice(all_agents), self.rng.choice(all_actions), self.rng.choice(all_objects), target_frame.tense, target_frame.polarity),
        ]

        # Generate support examples
        supp_id = 1
        for agent_c, act_c, obj_c, tense_c, pol_c in critical_components:
            if len(support_examples) >= k_support:
                break
            frame_s = EventFrame(agent=agent_c, action=act_c, object=obj_c, tense=tense_c, polarity=pol_c)
            utt_s, refs_s = grammar.generate_sentence(frame_s)
            frame_s.resolved_references = refs_s
            support_examples.append(SupportExample(
                example_id=supp_id,
                utterance=utt_s,
                event_frame=frame_s,
                context=[]
            ))
            supp_id += 1

        # Fill remaining support slots up to k_support
        while len(support_examples) < k_support:
            frame_s = EventFrame(
                agent=self.rng.choice(all_agents),
                action=self.rng.choice(all_actions),
                object=self.rng.choice(all_objects),
                tense=self.rng.choice(all_tenses),
                polarity=self.rng.choice(all_polarities)
            )
            utt_s, refs_s = grammar.generate_sentence(frame_s)
            frame_s.resolved_references = refs_s
            support_examples.append(SupportExample(
                example_id=supp_id,
                utterance=utt_s,
                event_frame=frame_s,
                context=[]
            ))
            supp_id += 1

        # Also, if query has dialogue phenomenon (e.g. coreference or ellipsis),
        # include at least 1 demonstration of that phenomenon in the support set!
        # Place it at the end of the support set to preserve critical lexical coverage at indices 0-3.
        if k_support >= 5 and phenomenon in ("coreference", "ellipsis", "correction"):
            supp_ctx, supp_q = dialogue_gen.generate_dialogue_episode(phenomenon=phenomenon)
            demo_idx = len(support_examples) - 1
            support_examples[demo_idx] = SupportExample(
                example_id=demo_idx + 1,
                utterance=supp_q.utterance,
                event_frame=supp_q.event_frame,
                context=[t.to_dict() for t in supp_ctx]
            )

        query_inst = QueryInstance(
            utterance=target_utterance,
            context=target_context,
            phenomenon=phenomenon
        )

        drift_attr = drift.get_frame_drift_attribution(target_frame)
        target_frame.drift_attribution = drift_attr

        # Determine formal IOAI competition difficulty level (Levels 1 to 5)
        if forced_difficulty_level is not None:
            diff_level = forced_difficulty_level
        elif forced_action is not None or forced_object is not None:
            # Compositional generalization query
            diff_level = DifficultyLevel.LEVEL_5_COMPOSITIONAL.value
        elif phenomenon in ("coreference", "ellipsis", "correction"):
            # Discourse reasoning query
            diff_level = DifficultyLevel.LEVEL_4_DISCOURSE.value
        elif target_frame.polarity == Polarity.NEGATIVE.value or any(v == "interaction_shift" for v in drift_attr.values()):
            # Pragmatic negation / polar inversion
            diff_level = DifficultyLevel.LEVEL_3_PRAGMATIC.value
        elif any(v in ("community_shift", "historical_shift") for v in drift_attr.values()) or period_id != HistoricalPeriod.CLASSICAL.value or community_id != CommunityId.CITADEL.value:
            # Dialectal and historical drift
            diff_level = DifficultyLevel.LEVEL_2_DRIFT.value
        else:
            # Canonical direct foundation
            diff_level = DifficultyLevel.LEVEL_1_DIRECT.value

        ep = Episode(
            episode_id=episode_id,
            community_id=community_id,
            period_id=period_id,
            word_order=drift.get_word_order(),
            support_set=support_examples,
            query=query_inst,
            query_label=target_frame,
            drift_attribution=drift_attr,
            difficulty_level=diff_level,
            metadata={
                "community_name": drift.community_profile.name,
                "period_name": drift.historical_profile.name,
                "negation_style": drift.get_negation_style(),
                "k_support": k_support,
                "phenomenon": phenomenon
            }
        )

        # Automatic Learnability Quality Gate & In-Place Repair
        from shift.learnability_validator import LearnabilityValidator
        audit = LearnabilityValidator.audit_episode(ep)
        if not audit.is_solvable:
            repaired_ep, success, _ = LearnabilityValidator.repair_episode(ep, grammar_engine=grammar)
            if success:
                ep = repaired_ep

        return ep

    def generate_counterfactual_episode_pair(
        self,
        pair_id: str,
        community_id: str = CommunityId.CITADEL.value,
        period_id: str = HistoricalPeriod.CLASSICAL.value,
        k_support: int = 8
    ) -> Tuple[Episode, Episode]:
        """
        Builds a paired counterfactual test instance.
        Episode A and Episode B share identical environment metadata, identical support set,
        and identical query utterance U_q, but have different dialogue contexts leading to different EventFrames.
        """
        ep_seed = deterministic_seed(f"{self.base_seed}_{pair_id}_{community_id}_{period_id}")
        base_lex = Lexicon(seed=ep_seed)
        drift = DriftEngine(community_id=community_id, period_id=period_id)
        drifted_lex = drift.transform_lexicon(base_lex)
        grammar = GrammarEngine(drifted_lex, drift.community_profile, drift.historical_profile)
        dialogue_gen = DialogueGenerator(grammar=grammar, seed=ep_seed)

        (ctx_a, q_a), (ctx_b, q_b) = dialogue_gen.generate_counterfactual_pair()

        # Build support set
        support_examples: List[SupportExample] = []
        all_agents = [a.value for a in AgentEntity if a not in (AgentEntity.SPEAKER, AgentEntity.ADDRESSEE)]
        all_actions = [a.value for a in ActionPredicate]
        all_objects = [o.value for o in ObjectEntity]
        all_tenses = [t.value for t in Tense]

        # Add coverage for both queries
        for frame in [q_a.event_frame, q_b.event_frame]:
            frame_s = EventFrame(agent=frame.agent, action=frame.action, object=frame.object, tense=frame.tense, polarity=Polarity.AFFIRMATIVE.value)
            utt_s, refs_s = grammar.generate_sentence(frame_s)
            frame_s.resolved_references = refs_s
            support_examples.append(SupportExample(
                example_id=len(support_examples)+1,
                utterance=utt_s,
                event_frame=frame_s
            ))

        while len(support_examples) < k_support:
            frame_s = EventFrame(
                agent=self.rng.choice(all_agents),
                action=self.rng.choice(all_actions),
                object=self.rng.choice(all_objects),
                tense=self.rng.choice(all_tenses),
                polarity=Polarity.AFFIRMATIVE.value
            )
            utt_s, refs_s = grammar.generate_sentence(frame_s)
            frame_s.resolved_references = refs_s
            support_examples.append(SupportExample(
                example_id=len(support_examples)+1,
                utterance=utt_s,
                event_frame=frame_s
            ))

        drift_attr_a = drift.get_frame_drift_attribution(q_a.event_frame)
        q_a.event_frame.drift_attribution = drift_attr_a

        drift_attr_b = drift.get_frame_drift_attribution(q_b.event_frame)
        q_b.event_frame.drift_attribution = drift_attr_b

        ep_a = Episode(
            episode_id=f"{pair_id}_A",
            community_id=community_id,
            period_id=period_id,
            word_order=drift.get_word_order(),
            support_set=support_examples,
            query=QueryInstance(utterance=q_a.utterance, context=[t.to_dict() for t in ctx_a], phenomenon="counterfactual_pair"),
            query_label=q_a.event_frame,
            drift_attribution=drift_attr_a,
            counterfactual_pair_id=pair_id,
            difficulty_level=5,
            metadata={"counterfactual_branch": "A"}
        )

        ep_b = Episode(
            episode_id=f"{pair_id}_B",
            community_id=community_id,
            period_id=period_id,
            word_order=drift.get_word_order(),
            support_set=support_examples,
            query=QueryInstance(utterance=q_b.utterance, context=[t.to_dict() for t in ctx_b], phenomenon="counterfactual_pair"),
            query_label=q_b.event_frame,
            drift_attribution=drift_attr_b,
            counterfactual_pair_id=pair_id,
            difficulty_level=5,
            metadata={"counterfactual_branch": "B"}
        )

        return ep_a, ep_b

    def generate_batch(
        self,
        count: int,
        k_support: int = 8,
        id_prefix: str = "ep"
    ) -> List[Episode]:
        """Generates a diverse batch of episodes across communities and periods."""
        communities = [c.value for c in CommunityId]
        periods = [p.value for p in HistoricalPeriod]
        phenomena = ["direct", "coreference", "ellipsis", "correction"]

        episodes: List[Episode] = []
        for i in range(count):
            ep_id = f"{id_prefix}_{i+1:04d}"
            comm = communities[i % len(communities)]
            period = periods[(i // len(communities)) % len(periods)]
            phenom = phenomena[i % len(phenomena)]

            ep = self.generate_single_episode(
                episode_id=ep_id,
                community_id=comm,
                period_id=period,
                k_support=k_support,
                phenomenon=phenom
            )
            episodes.append(ep)

        # Audit and guarantee 100% learnability across the entire batch (for k > 0)
        if k_support > 0:
            from shift.learnability_validator import LearnabilityValidator
            certified_episodes, _ = LearnabilityValidator.audit_and_repair_dataset(episodes, generator=self)
            return certified_episodes
        return episodes
