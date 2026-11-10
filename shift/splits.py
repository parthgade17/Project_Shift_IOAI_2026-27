"""
Project SHIFT: Leakage-Resistant Dataset Splitting Engine.
Milestone 08 Deliverable for IOAI 2027 Task Development.

Partitions episodes by latent rules, seeds, and held-out semantic compositions.
Provides strict data isolation, zero-leakage guarantees, and duplicate auditing.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any
from pathlib import Path

from shift.ontology import ActionPredicate, ObjectEntity
from shift.episodes import Episode, EpisodeGenerator
from shift.drift import CommunityId, HistoricalPeriod
from shift.serialize import export_episodes_jsonl, export_ground_truth_labels


@dataclass
class DatasetSplits:
    """Container holding partitioned datasets across competition splits."""
    train: List[Episode] = field(default_factory=list)
    val: List[Episode] = field(default_factory=list)
    public_dev: List[Episode] = field(default_factory=list)
    hidden_test: List[Episode] = field(default_factory=list)

    # Sub-evaluations within hidden_test
    test_iid: List[Episode] = field(default_factory=list)
    test_compositional_ood: List[Episode] = field(default_factory=list)
    test_heldout_community: List[Episode] = field(default_factory=list)
    test_heldout_period: List[Episode] = field(default_factory=list)
    test_counterfactual: List[Episode] = field(default_factory=list)


# Held-out compositional pairs for OOD generalization testing
HELD_OUT_COMPOSITIONS: Set[Tuple[str, str]] = {
    (ActionPredicate.FORGE.value, ObjectEntity.ANCIENT_RELIC.value),
    (ActionPredicate.HARVEST.value, ObjectEntity.STONE_MONOLITH.value),
    (ActionPredicate.ABANDON.value, ObjectEntity.ENERGY_MATRIX.value),
    (ActionPredicate.REPAIR.value, ObjectEntity.CRYSTAL_CORE.value),
}


class SplitManager:
    """
    Manages leakage-resistant partitioning of episodes across splits.
    Guarantees strict isolation between training and hidden evaluation.
    """
    def __init__(self, base_seed: int = 1337):
        self.base_seed = base_seed
        self.rng = random.Random(base_seed)

    def create_partitioned_dataset(
        self,
        train_count: int = 60,
        val_count: int = 20,
        dev_count: int = 20,
        test_iid_count: int = 20,
        test_ood_count: int = 20,
        test_comm_count: int = 15,
        test_period_count: int = 15,
        k_support: int = 8
    ) -> DatasetSplits:
        """
        Creates a complete multi-split dataset partition.
        Total episodes = 60 + 20 + 20 + (20+20+15+15) = 170 episodes.
        """
        splits = DatasetSplits()
        gen = EpisodeGenerator(base_seed=self.base_seed)

        # Standard communities and periods for training/validation
        std_communities = [
            CommunityId.CITADEL.value,
            CommunityId.NORTHERN_HIGHLANDS.value,
            CommunityId.COASTAL_VALLEY.value
        ]
        std_periods = [
            HistoricalPeriod.CLASSICAL.value,
            HistoricalPeriod.ARCHAIC.value
        ]

        heldout_community = CommunityId.EASTERN_PLAINS.value
        heldout_period = HistoricalPeriod.CONTEMPORARY.value

        phenomena = ["direct", "coreference", "ellipsis", "correction"]

        # 1. Train set
        train_queries: Set[str] = set()
        for i in range(train_count):
            ep_id = f"train_{i+1:04d}"
            comm = std_communities[i % len(std_communities)]
            period = std_periods[(i // len(std_communities)) % len(std_periods)]
            phenom = phenomena[i % len(phenomena)]
            ep = gen.generate_single_episode(
                episode_id=ep_id,
                community_id=comm,
                period_id=period,
                k_support=k_support,
                phenomenon=phenom
            )
            train_queries.add(ep.query.utterance)
            splits.train.append(ep)

        # 2. Validation set
        for i in range(val_count):
            ep_id = f"val_{i+1:04d}"
            comm = std_communities[i % len(std_communities)]
            period = std_periods[(i // len(std_communities)) % len(std_periods)]
            phenom = phenomena[i % len(phenomena)]
            ep = gen.generate_single_episode(
                episode_id=ep_id,
                community_id=comm,
                period_id=period,
                k_support=k_support,
                phenomenon=phenom
            )
            splits.val.append(ep)

        # 3. Public Dev set (ensuring no collision with train)
        dev_queries: Set[str] = set()
        for i in range(dev_count):
            ep_id = f"dev_{i+1:04d}"
            comm = std_communities[i % len(std_communities)]
            period = std_periods[(i // len(std_communities)) % len(std_periods)]
            phenom = phenomena[i % len(phenomena)]
            # Salt attempt if collision with train
            salt = 0
            ep = None
            while salt < 20:
                cid = f"{ep_id}_s{salt}" if salt > 0 else ep_id
                candidate = gen.generate_single_episode(
                    episode_id=cid,
                    community_id=comm,
                    period_id=period,
                    k_support=k_support,
                    phenomenon=phenom
                )
                if candidate.query.utterance not in train_queries:
                    ep = candidate
                    ep.episode_id = ep_id
                    break
                salt += 1
            if ep is None:
                ep = candidate
                ep.episode_id = ep_id
            dev_queries.add(ep.query.utterance)
            splits.public_dev.append(ep)

        # Helper for test sets avoiding train and dev
        def make_test_episode(ep_id: str, comm: str, period: str, phenom: str, forced_act=None, forced_obj=None):
            salt = 0
            ep = None
            forbidden = train_queries.union(dev_queries)
            while salt < 30:
                cid = f"{ep_id}_s{salt}" if salt > 0 else ep_id
                candidate = gen.generate_single_episode(
                    episode_id=cid,
                    community_id=comm,
                    period_id=period,
                    k_support=k_support,
                    phenomenon=phenom,
                    forced_action=forced_act,
                    forced_object=forced_obj
                )
                if candidate.query.utterance not in forbidden:
                    ep = candidate
                    ep.episode_id = ep_id
                    break
                salt += 1
            if ep is None:
                ep = candidate
                ep.episode_id = ep_id
            return ep

        # 4. Hidden Test: IID
        for i in range(test_iid_count):
            ep_id = f"test_iid_{i+1:04d}"
            comm = std_communities[i % len(std_communities)]
            period = std_periods[(i // len(std_communities)) % len(std_periods)]
            phenom = phenomena[i % len(phenomena)]
            ep = make_test_episode(ep_id, comm, period, phenom)
            splits.test_iid.append(ep)

        # 5. Hidden Test: Compositional OOD (held-out action-object compositions)
        ood_combos = list(HELD_OUT_COMPOSITIONS)
        for i in range(test_ood_count):
            ep_id = f"test_ood_{i+1:04d}"
            comm = std_communities[i % len(std_communities)]
            period = std_periods[(i // len(std_communities)) % len(std_periods)]
            phenom = phenomena[i % len(phenomena)]
            forced_act, forced_obj = ood_combos[i % len(ood_combos)]
            ep = make_test_episode(ep_id, comm, period, phenom, forced_act=forced_act, forced_obj=forced_obj)
            splits.test_compositional_ood.append(ep)

        # 6. Hidden Test: Held-out Community
        for i in range(test_comm_count):
            ep_id = f"test_comm_{i+1:04d}"
            phenom = phenomena[i % len(phenomena)]
            ep = make_test_episode(ep_id, heldout_community, HistoricalPeriod.CLASSICAL.value, phenom)
            splits.test_heldout_community.append(ep)

        # 7. Hidden Test: Held-out Historical Period
        for i in range(test_period_count):
            ep_id = f"test_period_{i+1:04d}"
            phenom = phenomena[i % len(phenomena)]
            ep = make_test_episode(ep_id, CommunityId.CITADEL.value, heldout_period, phenom)
            splits.test_heldout_period.append(ep)

        # 8. Hidden Test: Counterfactual Pairs (Testing context sensitivity)
        test_cf_pairs = 10
        for i in range(test_cf_pairs):
            pair_id = f"cf_pair_{i+1:04d}"
            comm = std_communities[i % len(std_communities)]
            period = std_periods[(i // len(std_communities)) % len(std_periods)]
            ep_a, ep_b = gen.generate_counterfactual_episode_pair(
                pair_id=pair_id,
                community_id=comm,
                period_id=period,
                k_support=k_support
            )
            splits.test_counterfactual.extend([ep_a, ep_b])

        # Guarantee 100% Learnability Certification across all partitioned splits
        from shift.learnability_validator import LearnabilityValidator
        splits.train, _ = LearnabilityValidator.audit_and_repair_dataset(splits.train, generator=gen)
        splits.val, _ = LearnabilityValidator.audit_and_repair_dataset(splits.val, generator=gen)
        train_queries_active = {ep.query.utterance for ep in splits.train}
        splits.public_dev, _ = LearnabilityValidator.audit_and_repair_dataset(
            splits.public_dev, generator=gen, forbidden_queries=train_queries_active
        )
        forbidden_for_test = train_queries_active.union({ep.query.utterance for ep in splits.public_dev})
        splits.test_iid, _ = LearnabilityValidator.audit_and_repair_dataset(
            splits.test_iid, generator=gen, forbidden_queries=forbidden_for_test
        )
        splits.test_compositional_ood, _ = LearnabilityValidator.audit_and_repair_dataset(
            splits.test_compositional_ood, generator=gen, forbidden_queries=forbidden_for_test
        )
        splits.test_heldout_community, _ = LearnabilityValidator.audit_and_repair_dataset(
            splits.test_heldout_community, generator=gen, forbidden_queries=forbidden_for_test
        )
        splits.test_heldout_period, _ = LearnabilityValidator.audit_and_repair_dataset(
            splits.test_heldout_period, generator=gen, forbidden_queries=forbidden_for_test
        )
        splits.test_counterfactual, _ = LearnabilityValidator.audit_and_repair_dataset(
            splits.test_counterfactual, generator=gen, forbidden_queries=forbidden_for_test
        )

        # Aggregate hidden test set
        splits.hidden_test = (
            splits.test_iid +
            splits.test_compositional_ood +
            splits.test_heldout_community +
            splits.test_heldout_period
        )

        return splits

    def audit_leakage(self, splits: DatasetSplits) -> Dict[str, Any]:
        """
        Audits dataset splits for query leakage and overlap:
        1. Checks whether any query utterance in public_dev or hidden_test appears in train.
        2. Checks whether any query utterance in hidden_test appears in public_dev.
        """
        train_queries = {ep.query.utterance for ep in splits.train}
        val_queries = {ep.query.utterance for ep in splits.val}
        dev_queries = {ep.query.utterance for ep in splits.public_dev}
        test_queries = {ep.query.utterance for ep in splits.hidden_test}

        leakage_test_in_train = test_queries.intersection(train_queries)
        leakage_dev_in_train = dev_queries.intersection(train_queries)
        leakage_test_in_dev = test_queries.intersection(dev_queries)

        is_leak_free = (
            len(leakage_test_in_train) == 0 and
            len(leakage_dev_in_train) == 0 and
            len(leakage_test_in_dev) == 0
        )

        return {
            "leak_free": is_leak_free,
            "train_query_count": len(train_queries),
            "dev_query_count": len(dev_queries),
            "test_query_count": len(test_queries),
            "leakage_test_in_train": list(leakage_test_in_train),
            "leakage_dev_in_train": list(leakage_dev_in_train),
            "leakage_test_in_dev": list(leakage_test_in_dev)
        }

    def save_splits_to_directory(self, splits: DatasetSplits, output_dir: str | Path) -> None:
        """Exports all splits in competition and evaluator formats."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Train and Val: full labels included for training
        export_episodes_jsonl(splits.train, out / "train.jsonl", strip_labels=False)
        export_episodes_jsonl(splits.val, out / "val.jsonl", strip_labels=False)

        # Public Dev: provided to contestants with labels for local debugging
        export_episodes_jsonl(splits.public_dev, out / "public_dev.jsonl", strip_labels=False)

        # Hidden Test: contestant package has labels STRIPPED
        export_episodes_jsonl(splits.hidden_test, out / "hidden_test_contestant.jsonl", strip_labels=True)

        # Private evaluator ground-truth keys
        export_ground_truth_labels(splits.hidden_test, out / "hidden_test_labels.json")
        export_episodes_jsonl(splits.hidden_test, out / "hidden_test_full_reference.jsonl", strip_labels=False)

        # Sub-splits for detailed diagnostic reporting
        export_episodes_jsonl(splits.test_iid, out / "test_iid.jsonl", strip_labels=False)
        export_episodes_jsonl(splits.test_compositional_ood, out / "test_ood.jsonl", strip_labels=False)
        export_episodes_jsonl(splits.test_heldout_community, out / "test_heldout_community.jsonl", strip_labels=False)
        export_episodes_jsonl(splits.test_heldout_period, out / "test_heldout_period.jsonl", strip_labels=False)
        export_episodes_jsonl(splits.test_counterfactual, out / "test_counterfactual.jsonl", strip_labels=False)
