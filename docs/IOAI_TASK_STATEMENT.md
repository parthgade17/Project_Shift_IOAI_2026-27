# International Olympiad in Artificial Intelligence (IOAI 2027)
## Official Competition Task Statement: PROJECT SHIFT
### Track: Natural Language Processing & Computational Linguistics

---

### Task Metadata
- **Task Title**: SHIFT: Semantic Adaptation & Disentanglement Across Communities and Time
- **Target Audience**: High School Olympiad Contestants & Undergraduate AI Researchers
- **Evaluation Paradigm**: Autonomous Algorithmic Semantic Parsing
- **Execution Environment**: Offline Sandbox, Pure Python 3.10+ (Standard Library)
- **Time Limit per Episode**: 2.0 seconds
- **Memory Limit**: 2048 MB RAM
- **GPU Requirement**: None (CPU standard execution)

---

### 1. Problem Narrative & Scientific Context

In deep-space planetary surveys across the Kepler-442 system, automated sensor drones uncover hundreds of historical archives and radio transmissions between isolated human settlements. All settlements speak dialects derived from an ancestral standard tongue, but their linguistic conventions have diverged across two independent axes:

1. **Community Dialect Conventions**: Different regional colonies (e.g. *Northern Highlands*, *Coastal Valley*, *Eastern Plains*) evolved regional phonetic shifts, specialized vocabularies, and distinct word orders (SOV, SVO, VSO).
2. **Historical Diachronic Shift**: Across historical epochs (*Archaic*, *Classical*, *Contemporary*), linguistic morphology evolved from synthetic case-inflected forms (e.g. accusative suffix `-a` on patient objects) to analytic structures.
3. **Factorial Mutations**: In specific communities during specific eras, localized semantic changes occurred (e.g., an energy term shifted only in Coastal settlements during the Contemporary period).
4. **Conversational Multi-Turn Context**: Transmissions are rarely isolated sentences; they consist of conversational dialogues with pronouns (*"he"*, *"she"*, *"it"*), ellipsis (omitted repeated verbs), and conversational corrections (*"not X, but Y"*).

### Your Objective
Given a small labelled **Support Set** ($k \le 16$ examples) illustrating the local community's grammar and vocabulary, and an unlabelled **Conversational Query**, your system must predict the structured **Event Frame** representing what actually occurred.

---

### 2. Semantic World & Ontology Specification

Every communication event in Project SHIFT takes place within a closed semantic universe consisting of:

#### 2.1 Canonical Entities (Agents & Patients)
| Canonical ID | Entity Role | Pronoun Agreement | Description |
| :--- | :--- | :---: | :--- |
| `scout_alpha` | Scout | Masculine (`el`) | Lead reconnaissance scout |
| `scout_beta` | Scout | Feminine (`la`) | Sensor and mapping scout |
| `artisan_gamma` | Artisan | Masculine (`el`) | Mechanical and structural technician |
| `artisan_delta` | Artisan | Feminine (`la`) | Systems and circuit engineer |
| `scholar_omega` | Scholar | Masculine (`el`) | Chief archivist and linguist |
| `scholar_kappa` | Scholar | Feminine (`la`) | Astrobiology researcher |
| `guardian_theta` | Guardian | Masculine (`el`) | Perimeter security officer |
| `guardian_sigma` | Guardian | Feminine (`la`) | Life support defense officer |

#### 2.2 Canonical Actions (Predicates)
- `inspect`: Diagnostic or visual survey.
- `harvest`: Extracting or collecting resources.
- `forge`: Assembling, crafting, or building infrastructure.
- `secure`: Fortifying or protecting perimeter.
- `relocate`: Moving or transporting an object.
- `repair`: Fixing or restoring damaged components.
- `activate`: Initializing or powering on systems.
- `abandon`: Vacating or leaving behind.

#### 2.3 Canonical Objects
- `crystal_core`, `ancient_relic`, `signal_beacon`, `water_purifier`
- `stone_monolith`, `energy_matrix`, `data_capsule`, `shelter_dome`

#### 2.4 Grammatical Dimensions
- **Tense**: `past`, `present`, `future`
- **Polarity**: `affirmative`, `negative` (signaled by negation particles `non`, `sed`, or prefix `ne-`)
- **Word Order**: `SOV` (Subject-Object-Verb), `SVO` (Subject-Verb-Object), or `VSO` (Verb-Subject-Object)

---

### 3. Progressive Subtask Levels (Level 1 to Level 5)

| Level | Focus | Key Challenge |
| :---: | :--- | :--- |
| **Level 1** | **Direct Lexicon Induction** | Induce novel vocabulary in canonical word order with no dialect drift. |
| **Level 2** | **Single Drift Attribution** | Recognize single-source semantic changes (community dialect or historical era). |
| **Level 3** | **Factorial Pragmatics & Morphology** | Disentangle community $\times$ epoch interactions, archaic case affixes (`-a`), and negation. |
| **Level 4** | **Discourse Understanding** | Resolve pronominal anaphora, ellipsis (omitted verbs), and conversational repairs. |
| **Level 5** | **Compositional OOD & Counterfactuals** | Generalize to held-out action-object pairings and paired counterfactual context twins. |

---

### 4. Input & Output Data Format

#### 4.1 Contestant-Facing Input Episode (`.jsonl`)
Each line of the competition test file contains an episode JSON object with **zero answer keys**:

```json
{
  "episode_id": "ep_0105",
  "metadata": {
    "community": "Northern Highlands",
    "historical_period": "Contemporary",
    "word_order": "SOV",
    "difficulty_level": 4
  },
  "support_set": [
    {
      "utterance": "celde mon dynil.",
      "event_frame": {
        "agent": "artisan_delta",
        "action": "activate",
        "object": "stone_monolith",
        "tense": "past",
        "polarity": "affirmative"
      }
    },
    {
      "utterance": "morom hyd coren.",
      "event_frame": {
        "agent": "scholar_omega",
        "action": "harvest",
        "object": "water_purifier",
        "tense": "present",
        "polarity": "affirmative"
      }
    }
  ],
  "query": {
    "context": [
      {
        "speaker_id": "Speaker_A",
        "utterance": "celde mon dynil."
      }
    ],
    "utterance": "la id coren."
  }
}
```

> **Note on Zero Leakage**: Context turns contain only `speaker_id` and raw `utterance`. Contestants must parse the context utterance using the induced support vocabulary to discover antecedent entities.

#### 4.2 Submission Output Format
Contestants submit a single JSON file mapping each `episode_id` to its predicted `EventFrame`:

```json
{
  "ep_0105": {
    "agent": "artisan_delta",
    "action": "harvest",
    "object": "stone_monolith",
    "tense": "present",
    "polarity": "affirmative",
    "resolved_references": {
      "la": "artisan_delta",
      "id": "stone_monolith"
    }
  }
}
```

---

### 5. Official Scoring & Evaluation Metric

Submissions are scored on a scale from $0.0$ to $100.0$ using the official IOAI Composite Formula:

$$\text{Score} = 0.40 \cdot \text{Acc}_{\text{exact}} + 0.20 \cdot \text{F1}_{\text{macro}} + 0.15 \cdot \text{Acc}_{\text{disentangle}} + 0.15 \cdot \text{Acc}_{\text{ref\_res}} + 0.10 \cdot \text{Score}_{\text{counterfactual}}$$

Where:
- **$\text{Acc}_{\text{exact}}$**: Percentage of episodes where all 5 slots (`agent`, `action`, `object`, `tense`, `polarity`) exactly match the ground truth.
- **$\text{F1}_{\text{macro}}$**: Unweighted macro-average F1 score across all 5 frame slots.
- **$\text{Acc}_{\text{disentangle}}$**: Accuracy on classifying whether root morphemes shifted via community or historical drift.
- **$\text{Acc}_{\text{ref\_res}}$**: Accuracy on resolving discourse pronoun antecedents.
- **$\text{Score}_{\text{counterfactual}}$**: Harmonic mean of joint pair accuracy and perturbation sensitivity on counterfactual context twins.

---

### 6. Rules & Submission Guidelines

1. **Standard Library Only**: Submissions must execute using standard Python 3.10+ without importing external packages (e.g., `torch`, `transformers`, `numpy` are not permitted in offline evaluation).
2. **Deterministic Execution**: Submissions must produce identical outputs on repeated evaluations. Random seeds must be fixed.
3. **Offline Sandbox**: No internet access or external API calls are allowed during evaluation.
4. **Starter Code**: Contestants are provided with:
   - `loader.py`: Safe data loader with built-in zero-leakage assertions.
   - `submission_template.py`: Starter baseline template.
   - `local_evaluator.py`: Official local validator for offline self-evaluation against public dev data.
