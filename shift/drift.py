"""
Project SHIFT: Community and Historical Drift Engine.
Milestone 04 Deliverable for IOAI 2027 Task Development.

Implements a full Factorial Semantic Drift Framework:
- Community Main Effects: Dialectal shifts constant across epochs.
- Historical Main Effects: Temporal shifts constant across communities.
- Community x Historical Interaction Effects: Dialect-epoch specific semantic mutations.
- Drift Source Attribution for automated disentanglement testing.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import copy

from shift.ontology import ActionPredicate, ObjectEntity, Tense, Polarity, DriftSource, EventFrame
from shift.lexicon import Lexicon, MorphemeSet


class CommunityId(str, Enum):
    NORTHERN_HIGHLANDS = "comm_north"
    COASTAL_VALLEY = "comm_coast"
    EASTERN_PLAINS = "comm_east"
    CITADEL = "comm_citadel"


class HistoricalPeriod(str, Enum):
    ARCHAIC = "period_archaic"           # Epoch 0: Older inflectional form
    CLASSICAL = "period_classical"       # Epoch 1: Standard benchmark reference
    CONTEMPORARY = "period_contemporary" # Epoch 2: Analytic modern dialect


@dataclass
class CommunityProfile:
    """Defines community-specific dialect traits (Community Main Effects)."""
    community_id: str
    name: str
    phonological_shifts: Dict[str, str] = field(default_factory=dict)
    lexical_substitutions: Dict[str, str] = field(default_factory=dict) # concept -> new root
    negation_style: str = "prefix"  # "prefix" (ne-verb) or "particle" (verb non)
    word_order_preference: str = "SOV" # "SOV", "SVO", "VSO"


@dataclass
class HistoricalProfile:
    """Defines historical epoch shifts (Historical Main Effects)."""
    period_id: str
    name: str
    tense_affix_overrides: Dict[str, str] = field(default_factory=dict)
    semantic_drift_map: Dict[str, str] = field(default_factory=dict)
    use_case_marking: bool = False # Archaic synthetic case endings
    use_apocope: bool = False      # Contemporary terminal elision


COMMUNITY_CONFIGS: Dict[str, CommunityProfile] = {
    CommunityId.NORTHERN_HIGHLANDS.value: CommunityProfile(
        community_id=CommunityId.NORTHERN_HIGHLANDS.value,
        name="Northern Highlands",
        phonological_shifts={"k": "c", "t": "th"},
        lexical_substitutions={
            ActionPredicate.HARVEST.value: "kor",
            ObjectEntity.CRYSTAL_CORE.value: "glac",
        },
        negation_style="prefix",
        word_order_preference="SOV",
    ),
    CommunityId.COASTAL_VALLEY.value: CommunityProfile(
        community_id=CommunityId.COASTAL_VALLEY.value,
        name="Coastal Valley",
        phonological_shifts={"p": "f", "s": "sh"},
        lexical_substitutions={
            ActionPredicate.SECURE.value: "mar",
            ObjectEntity.WATER_PURIFIER.value: "aqu",
        },
        negation_style="particle",
        word_order_preference="SVO",
    ),
    CommunityId.EASTERN_PLAINS.value: CommunityProfile(
        community_id=CommunityId.EASTERN_PLAINS.value,
        name="Eastern Plains",
        phonological_shifts={"b": "v", "d": "z"},
        lexical_substitutions={
            ActionPredicate.FORGE.value: "smi",
            ObjectEntity.SIGNAL_BEACON.value: "luz",
        },
        negation_style="prefix",
        word_order_preference="VSO",
    ),
    CommunityId.CITADEL.value: CommunityProfile(
        community_id=CommunityId.CITADEL.value,
        name="The Citadel",
        phonological_shifts={}, # Baseline standard ancestral reference
        lexical_substitutions={},
        negation_style="prefix",
        word_order_preference="SOV",
    ),
}

HISTORICAL_CONFIGS: Dict[str, HistoricalProfile] = {
    HistoricalPeriod.ARCHAIC.value: HistoricalProfile(
        period_id=HistoricalPeriod.ARCHAIC.value,
        name="Archaic Epoch",
        tense_affix_overrides={
            Tense.PAST.value: "ost",
            Tense.PRESENT.value: "ant",
            Tense.FUTURE.value: "esc",
        },
        use_case_marking=True, # Appends nominative -a, accusative -om
        use_apocope=False,
    ),
    HistoricalPeriod.CLASSICAL.value: HistoricalProfile(
        period_id=HistoricalPeriod.CLASSICAL.value,
        name="Classical Epoch",
        tense_affix_overrides={
            Tense.PAST.value: "il",
            Tense.PRESENT.value: "en",
            Tense.FUTURE.value: "or",
        },
        use_case_marking=False,
        use_apocope=False,
    ),
    HistoricalPeriod.CONTEMPORARY.value: HistoricalProfile(
        period_id=HistoricalPeriod.CONTEMPORARY.value,
        name="Contemporary Epoch",
        tense_affix_overrides={
            Tense.PAST.value: "ed",
            Tense.PRESENT.value: "in",
            Tense.FUTURE.value: "az",
        },
        use_case_marking=False,
        use_apocope=True,
    ),
}

# Factorial Interaction Effects: (community_id, period_id) -> {concept: localized_shifted_root}
FACTORIAL_INTERACTIONS: Dict[Tuple[str, str], Dict[str, str]] = {
    # Coastal Valley in Contemporary Epoch evolved specialized slang for crystal core: 'lumis'
    (CommunityId.COASTAL_VALLEY.value, HistoricalPeriod.CONTEMPORARY.value): {
        ObjectEntity.CRYSTAL_CORE.value: "lumis",
    },
    # Northern Highlands in Archaic Epoch evolved a distinct term for secure: 'firm'
    (CommunityId.NORTHERN_HIGHLANDS.value, HistoricalPeriod.ARCHAIC.value): {
        ActionPredicate.SECURE.value: "firm",
    },
    # Eastern Plains in Contemporary Epoch evolved localized slang for signal beacon: 'radi'
    (CommunityId.EASTERN_PLAINS.value, HistoricalPeriod.CONTEMPORARY.value): {
        ObjectEntity.SIGNAL_BEACON.value: "radi",
    },
}


class DriftEngine:
    """
    Applies community dialect shifts and historical time evolutions to a base Lexicon.
    Produces a specialized Lexicon instance specific to (community, historical_period)
    and enables semantic change disentanglement.
    """
    def __init__(
        self,
        community_id: str = CommunityId.CITADEL.value,
        period_id: str = HistoricalPeriod.CLASSICAL.value
    ):
        self.community_id = community_id
        self.period_id = period_id
        self.community_profile = COMMUNITY_CONFIGS.get(
            community_id,
            COMMUNITY_CONFIGS[CommunityId.CITADEL.value]
        )
        self.historical_profile = HISTORICAL_CONFIGS.get(
            period_id,
            HISTORICAL_CONFIGS[HistoricalPeriod.CLASSICAL.value]
        )

    def apply_phonological_shift(self, text: str) -> str:
        """Applies sound shifts defined for the community."""
        transformed = text
        for src, tgt in self.community_profile.phonological_shifts.items():
            transformed = transformed.replace(src, tgt)
        return transformed

    def transform_lexicon(self, base_lexicon: Lexicon) -> Lexicon:
        """
        Derives an evolved Lexicon from base lexicon according to:
        1. Historical Main Effects
        2. Community Main Effects
        3. Factorial Interaction Effects
        4. Phonological shifts
        """
        morphemes = copy.deepcopy(base_lexicon.morphemes)

        # 1. Historical tense affix overrides (Historical Main Effect)
        for tense_val, affix in self.historical_profile.tense_affix_overrides.items():
            morphemes.tense_affixes[tense_val] = affix

        # 2. Community lexical substitutions (Community Main Effect)
        for concept, sub_root in self.community_profile.lexical_substitutions.items():
            if concept in morphemes.action_roots:
                morphemes.action_roots[concept] = sub_root
            elif concept in morphemes.object_roots:
                morphemes.object_roots[concept] = sub_root

        # 3. Factorial Interaction overrides (Interaction Effect)
        interaction_key = (self.community_id, self.period_id)
        if interaction_key in FACTORIAL_INTERACTIONS:
            for concept, sub_root in FACTORIAL_INTERACTIONS[interaction_key].items():
                if concept in morphemes.action_roots:
                    morphemes.action_roots[concept] = sub_root
                elif concept in morphemes.object_roots:
                    morphemes.object_roots[concept] = sub_root

        # 4. Apply phonological shifts across morpheme tables
        for k in morphemes.role_roots:
            morphemes.role_roots[k] = self.apply_phonological_shift(morphemes.role_roots[k])
        for k in morphemes.entity_suffixes:
            morphemes.entity_suffixes[k] = self.apply_phonological_shift(morphemes.entity_suffixes[k])
        for k in morphemes.action_roots:
            morphemes.action_roots[k] = self.apply_phonological_shift(morphemes.action_roots[k])
        for k in morphemes.object_roots:
            morphemes.object_roots[k] = self.apply_phonological_shift(morphemes.object_roots[k])
        for k in morphemes.tense_affixes:
            morphemes.tense_affixes[k] = self.apply_phonological_shift(morphemes.tense_affixes[k])
        for k in morphemes.pronoun_roots:
            morphemes.pronoun_roots[k] = self.apply_phonological_shift(morphemes.pronoun_roots[k])

        morphemes.determiner = self.apply_phonological_shift(morphemes.determiner)
        morphemes.conjunction_and = self.apply_phonological_shift(morphemes.conjunction_and)
        morphemes.negation_particle = self.apply_phonological_shift(morphemes.negation_particle)
        morphemes.correction_particle = self.apply_phonological_shift(morphemes.correction_particle)

        # Build newly drifted lexicon
        drifted_lex = Lexicon(morphemes=morphemes, seed=base_lexicon.seed)
        return drifted_lex

    def get_word_order(self) -> str:
        """Returns the word order (e.g. 'SOV', 'SVO', 'VSO') for this community/period."""
        return self.community_profile.word_order_preference

    def get_negation_style(self) -> str:
        """Returns 'prefix' or 'particle'."""
        return self.community_profile.negation_style

    def get_drift_attribution(self, concept: str) -> str:
        """
        Disentanglement oracle:
        Determines whether meaning changed due to community, historical time, or their interaction.
        """
        interaction_key = (self.community_id, self.period_id)
        if interaction_key in FACTORIAL_INTERACTIONS and concept in FACTORIAL_INTERACTIONS[interaction_key]:
            return DriftSource.INTERACTION_SHIFT.value

        if concept in self.community_profile.lexical_substitutions:
            return DriftSource.COMMUNITY_SHIFT.value

        if concept in (Tense.PAST.value, Tense.PRESENT.value, Tense.FUTURE.value):
            if self.period_id != HistoricalPeriod.CLASSICAL.value:
                return DriftSource.HISTORICAL_SHIFT.value
            return DriftSource.STABLE.value

        if concept in self.historical_profile.semantic_drift_map:
            return DriftSource.HISTORICAL_SHIFT.value

        return DriftSource.STABLE.value

    def get_frame_drift_attribution(self, frame: EventFrame) -> Dict[str, str]:
        """Calculates ground-truth attribution for all slots in an EventFrame."""
        return {
            "agent": self.get_drift_attribution(frame.agent),
            "action": self.get_drift_attribution(frame.action),
            "object": self.get_drift_attribution(frame.object),
            "tense": self.get_drift_attribution(frame.tense),
            "polarity": DriftSource.STABLE.value
        }
