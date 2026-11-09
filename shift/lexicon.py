"""
Project SHIFT: Compositional Language and Lexicon Generator.
Milestone 03 Deliverable for IOAI 2027 Task Development.

Generates base morphemes, systematic compound word formation, detects
accidental collisions, and provides forward/reverse vocabulary mappings.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple, Any

from shift.ontology import (
    AgentEntity, AgentRole, ActionPredicate, ObjectEntity,
    Tense, Polarity, AGENT_METADATA
)


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes standard edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


@dataclass
class MorphemeSet:
    """Canonical collection of base roots, affixes, and functional particles."""
    # Role roots (Agent category)
    role_roots: Dict[str, str] = field(default_factory=lambda: {
        AgentRole.SCOUT.value: "tor",
        AgentRole.ARTISAN.value: "kel",
        AgentRole.SCHOLAR.value: "mor",
        AgentRole.GUARDIAN.value: "zar",
    })

    # Entity suffix identifiers
    entity_suffixes: Dict[str, str] = field(default_factory=lambda: {
        "alpha": "al",
        "beta": "be",
        "gamma": "ga",
        "delta": "de",
        "omega": "om",
        "kappa": "ka",
        "theta": "th",
        "sigma": "si",
    })

    # Action predicates
    action_roots: Dict[str, str] = field(default_factory=lambda: {
        ActionPredicate.INSPECT.value: "vid",
        ActionPredicate.HARVEST.value: "fen",
        ActionPredicate.FORGE.value: "vok",
        ActionPredicate.SECURE.value: "rak",
        ActionPredicate.RELOCATE.value: "lum",
        ActionPredicate.REPAIR.value: "san",
        ActionPredicate.ACTIVATE.value: "dyn",
        ActionPredicate.ABANDON.value: "nul",
    })

    # Object entities
    object_roots: Dict[str, str] = field(default_factory=lambda: {
        ObjectEntity.CRYSTAL_CORE.value: "kris",
        ObjectEntity.ANCIENT_RELIC.value: "ark",
        ObjectEntity.SIGNAL_BEACON.value: "bek",
        ObjectEntity.WATER_PURIFIER.value: "hyd",
        ObjectEntity.STONE_MONOLITH.value: "mon",
        ObjectEntity.ENERGY_MATRIX.value: "mat",
        ObjectEntity.DATA_CAPSULE.value: "dox",
        ObjectEntity.SHELTER_DOME.value: "dom",
    })

    # Tense markers
    tense_affixes: Dict[str, str] = field(default_factory=lambda: {
        Tense.PAST.value: "il",
        Tense.PRESENT.value: "en",
        Tense.FUTURE.value: "or",
    })

    # Polarity markers
    polarity_affixes: Dict[str, str] = field(default_factory=lambda: {
        Polarity.AFFIRMATIVE.value: "",
        Polarity.NEGATIVE.value: "ne",
    })

    # Referring expressions / pronouns
    pronoun_roots: Dict[str, str] = field(default_factory=lambda: {
        "speaker": "mi",         # 1st person ("I")
        "addressee": "tu",       # 2nd person ("you")
        "masculine": "el",       # 3rd person masc ("he")
        "feminine": "la",        # 3rd person fem ("she")
        "neuter": "id",          # 3rd person neuter ("it")
        "demonstrative": "se",   # demonstrative ("that")
    })

    # Functional particles
    determiner: str = "te"       # "the"
    conjunction_and: str = "et"  # "and"
    negation_particle: str = "non"# freestanding negative particle
    correction_particle: str = "sed"# "but / rather / instead"


class Lexicon:
    """
    Manages the bidirectional vocabulary mapping for a controlled language instance.
    Handles systematic compounding, collision detection, and ambiguous mapping prevention.
    """
    def __init__(self, morphemes: Optional[MorphemeSet] = None, seed: Optional[int] = None):
        self.seed = seed
        self.morphemes = morphemes or MorphemeSet()
        self.concept_to_surface: Dict[str, str] = {}
        self.surface_to_concept: Dict[str, str] = {}
        self._build_lexicon()
        self.audit_collisions()

    def _build_lexicon(self) -> None:
        """Construct full compositional words from morphemes."""
        # 1. Agent entity words (Role + Suffix)
        entity_map = {
            AgentEntity.SCOUT_ALPHA.value: (AgentRole.SCOUT.value, "alpha"),
            AgentEntity.SCOUT_BETA.value: (AgentRole.SCOUT.value, "beta"),
            AgentEntity.ARTISAN_GAMMA.value: (AgentRole.ARTISAN.value, "gamma"),
            AgentEntity.ARTISAN_DELTA.value: (AgentRole.ARTISAN.value, "delta"),
            AgentEntity.SCHOLAR_OMEGA.value: (AgentRole.SCHOLAR.value, "omega"),
            AgentEntity.SCHOLAR_KAPPA.value: (AgentRole.SCHOLAR.value, "kappa"),
            AgentEntity.GUARDIAN_THETA.value: (AgentRole.GUARDIAN.value, "theta"),
            AgentEntity.GUARDIAN_SIGMA.value: (AgentRole.GUARDIAN.value, "sigma"),
            AgentEntity.SPEAKER.value: ("speaker", None),
            AgentEntity.ADDRESSEE.value: ("addressee", None),
        }

        for agent_val, (role_or_type, suffix_key) in entity_map.items():
            if suffix_key is not None:
                role_root = self.morphemes.role_roots[role_or_type]
                suffix = self.morphemes.entity_suffixes[suffix_key]
                surface = f"{role_root}{suffix}"
            else:
                surface = self.morphemes.pronoun_roots[role_or_type]
            self._register(agent_val, surface)

        # 2. Action verbs (base roots)
        for action_val, root in self.morphemes.action_roots.items():
            self._register(action_val, root)

        # 3. Object nouns (base roots)
        for obj_val, root in self.morphemes.object_roots.items():
            self._register(obj_val, root)

        # 4. Tense markers
        for tense_val, affix in self.morphemes.tense_affixes.items():
            self._register(f"tense:{tense_val}", affix)

        # 5. Polarity markers
        for pol_val, affix in self.morphemes.polarity_affixes.items():
            if affix:
                self._register(f"polarity:{pol_val}", affix)

        # 6. Functional tokens
        self._register("det:te", self.morphemes.determiner)
        self._register("conj:et", self.morphemes.conjunction_and)
        self._register("neg:non", self.morphemes.negation_particle)
        self._register("corr:sed", self.morphemes.correction_particle)

        # 7. Pronouns (excluding speaker and addressee which are already registered as Agent entities)
        for pron_key, root in self.morphemes.pronoun_roots.items():
            if pron_key not in ("speaker", "addressee"):
                self._register(f"pronoun:{pron_key}", root)

    def _register(self, concept: str, surface: str) -> None:
        self.concept_to_surface[concept] = surface
        self.surface_to_concept[surface] = concept

    def audit_collisions(self, min_edit_distance: int = 1) -> List[str]:
        """
        Detects collisions or dangerous orthographic ambiguities.
        Throws ValueError if exact duplicate surfaces map to different concepts.
        """
        warnings: List[str] = []
        surfaces = list(self.surface_to_concept.keys())

        # Exact collisions check
        surface_counts: Dict[str, List[str]] = {}
        for c, s in self.concept_to_surface.items():
            surface_counts.setdefault(s, []).append(c)

        for s, concepts in surface_counts.items():
            if len(concepts) > 1:
                raise ValueError(
                    f"Fatal lexicon collision: surface '{s}' mapped to multiple concepts: {concepts}"
                )

        # Edit distance proximity audit
        for i in range(len(surfaces)):
            for j in range(i + 1, len(surfaces)):
                s1, s2 = surfaces[i], surfaces[j]
                dist = levenshtein_distance(s1, s2)
                if dist < min_edit_distance:
                    warnings.append(f"Near-collision: '{s1}' vs '{s2}' (dist={dist})")

        return warnings

    def get_surface(self, concept: str) -> str:
        """Fetch surface word for a canonical concept."""
        if concept == "pronoun:speaker":
            concept = AgentEntity.SPEAKER.value
        elif concept == "pronoun:addressee":
            concept = AgentEntity.ADDRESSEE.value

        if concept not in self.concept_to_surface:
            raise KeyError(f"Concept '{concept}' not in lexicon")
        return self.concept_to_surface[concept]

    def get_concept(self, surface: str) -> Optional[str]:
        """Lookup semantic concept from surface word."""
        return self.surface_to_concept.get(surface)

    def inflect_verb(self, action: str, tense: str, polarity: str, synthetic: bool = True) -> str:
        """
        Realise inflected verb form.
        If synthetic: root + tense suffix (with optional ne- prefix).
        If analytic: auxiliary/particle + root.
        """
        root = self.get_surface(action)
        tense_affix = self.morphemes.tense_affixes[tense]

        if synthetic:
            verb_form = f"{root}{tense_affix}"
            if polarity == Polarity.NEGATIVE.value:
                verb_form = f"{self.morphemes.polarity_affixes[Polarity.NEGATIVE.value]}{verb_form}"
            return verb_form
        else:
            # Analytic morphology: 'non' [optional] + root + tense particle
            parts = []
            if polarity == Polarity.NEGATIVE.value:
                parts.append(self.morphemes.negation_particle)
            parts.append(root)
            parts.append(tense_affix)
            return " ".join(parts)


def generate_random_lexicon(seed: int) -> Lexicon:
    """Generates a randomized, collision-safe dialect lexicon using pseudo-phonology."""
    rng = random.Random(seed)
    consonants = ["p", "t", "k", "b", "d", "g", "m", "n", "s", "z", "f", "v", "l", "r"]
    vowels = ["a", "e", "i", "o", "u"]

    used_surfaces: Set[str] = set()

    def generate_syllable(pattern: str = "CVC") -> str:
        for _ in range(100):
            chars = []
            for char_type in pattern:
                if char_type == "C":
                    chars.append(rng.choice(consonants))
                elif char_type == "V":
                    chars.append(rng.choice(vowels))
            s = "".join(chars)
            if s not in used_surfaces:
                used_surfaces.add(s)
                return s
        raise RuntimeError("Failed to generate unique syllable; exhaustion")

    morphemes = MorphemeSet(
        role_roots={
            AgentRole.SCOUT.value: generate_syllable("CVC"),
            AgentRole.ARTISAN.value: generate_syllable("CVC"),
            AgentRole.SCHOLAR.value: generate_syllable("CVC"),
            AgentRole.GUARDIAN.value: generate_syllable("CVC"),
        },
        entity_suffixes={
            "alpha": generate_syllable("VC"),
            "beta": generate_syllable("VC"),
            "gamma": generate_syllable("VC"),
            "delta": generate_syllable("VC"),
            "omega": generate_syllable("VC"),
            "kappa": generate_syllable("VC"),
            "theta": generate_syllable("VC"),
            "sigma": generate_syllable("VC"),
        },
        action_roots={action.value: generate_syllable("CVC") for action in ActionPredicate},
        object_roots={obj.value: generate_syllable("CVC") for obj in ObjectEntity},
        tense_affixes={
            Tense.PAST.value: generate_syllable("VC"),
            Tense.PRESENT.value: generate_syllable("VC"),
            Tense.FUTURE.value: generate_syllable("VC"),
        },
        polarity_affixes={
            Polarity.AFFIRMATIVE.value: "",
            Polarity.NEGATIVE.value: generate_syllable("CV"),
        },
        pronoun_roots={
            "speaker": generate_syllable("CV"),
            "addressee": generate_syllable("CV"),
            "masculine": generate_syllable("VC"),
            "feminine": generate_syllable("VC"),
            "neuter": generate_syllable("VC"),
            "demonstrative": generate_syllable("CV"),
        },
        determiner=generate_syllable("CV"),
        conjunction_and=generate_syllable("VC"),
        negation_particle=generate_syllable("CVC"),
        correction_particle=generate_syllable("CVC"),
    )

    return Lexicon(morphemes=morphemes, seed=seed)
