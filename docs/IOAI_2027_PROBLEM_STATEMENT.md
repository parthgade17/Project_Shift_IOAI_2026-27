# IOAI 2027 Contest Problem Statement
## Task: PROJECT SHIFT (Semantic Adaptation & Disentanglement Across Communities and Time)

**Task Category**: Natural Language Processing & Computational Linguistics  
**Target Audience**: High School Olympiad Contestants & Undergraduate AI Students  
**Evaluation Mode**: Autonomous Algorithmic & Model Evaluation  
**Time Limit per Episode**: 2.0 seconds  
**Memory Limit**: 2048 MB  

---

### 1. Problem Narrative & The Core Scientific Challenge

In the year 2340, deep-space planetary surveys across the Kepler-442 system uncover hundreds of historical archives and field transmissions from isolated human colony settlements. While all settlements descended from a common ancestral language, linguistic conventions have evolved through two distinct forces:
1. **Community-Driven Drift**: Distinct geographical communities (e.g. Northern Highlands vs Coastal Valley vs Eastern Plains) evolved regional sound changes, lexical substitutions, and word orders (SOV, SVO, VSO).
2. **Historical-Time Drift**: Across historical epochs (Archaic, Classical, Contemporary), grammatical systems shifted from synthetic case-inflected forms to analytic auxiliary particles.
3. **Factorial Interactions**: In specific communities during specific eras, localized semantic mutations occurred (e.g., an interaction effect where an energy term shifted only in Coastal settlements during the Contemporary era).
4. **Conversational Context**: Transmissions are multi-turn dialogues with coreference (*"he"*, *"she"*, *"it"*), ellipsis, and conversational repairs (*"not X, but Y"*).

### The Contestant's Objective:
Your AI system must:
1. Infer the community's vocabulary and grammar from a small labelled support set ($k \le 16$).
2. Resolve multi-turn conversational context (anaphora, ellipsis, repairs).
3. Predict the canonical structured **Event Frame**.
4. **Disentangle Semantic Change**: Attribute whether words changed meaning due to `community_shift`, `historical_shift`, `interaction_shift`, or remained `stable`.

---

### 2. Semantic World & Ontology

#### 2.1 Canonical Agents
- **Scouts**: `scout_alpha` (masculine, agreement pronoun `el`), `scout_beta` (feminine, agreement pronoun `la`)
- **Artisans**: `artisan_gamma` (masculine), `artisan_delta` (feminine)
- **Scholars**: `scholar_omega` (masculine), `scholar_kappa` (feminine)
- **Guardians**: `guardian_theta` (masculine), `guardian_sigma` (feminine)
- **Participants**: `speaker` (1st person `mi`), `addressee` (2nd person `tu`)

#### 2.2 Canonical Actions
- `inspect`: visual, sensor, or diagnostic scan.
- `harvest`: gathering, extracting, or collecting resources.
- `forge`: crafting, engineering, or assembling infrastructure.
- `secure`: fortifying, protecting, or establishing perimeter defense.
- `relocate`: transporting or moving an object to another site.
- `repair`: fixing or restoring damaged systems.
- `activate`: initializing or powering on a device.
- `abandon`: vacating or leaving behind an artifact or structure.

#### 2.3 Canonical Objects
- `crystal_core`, `ancient_relic`, `signal_beacon`, `water_purifier`
- `stone_monolith`, `energy_matrix`, `data_capsule`, `shelter_dome`

#### 2.4 Tense, Polarity, and Drift Sources
- **Tense**: `past`, `present`, `future`
- **Polarity**: `affirmative`, `negative`
- **Drift Sources**:
  - `stable`: inherited from standard ancestral language unchanged.
  - `community_shift`: shifted due to community dialect convention across all epochs.
  - `historical_shift`: shifted due to epoch time across all communities.
  - `interaction_shift`: shifted due to specific community $\times$ epoch interaction.

---

### 3. Progressive Subtask Levels (Level 1 to Level 5)

To provide educational value and fair discrimination, the competition features 5 progressive levels:

| Level | Subtask Focus | What Contestants Must Do |
| :---: | :--- | :--- |
| **Level 1** | **Direct Lexicon Induction** | Map unfamiliar vocabulary in standard word order without dialect drift. |
| **Level 2** | **Single Drift Attribution** | Recognize single-source semantic changes (community-only or historical-only). |
| **Level 3** | **Factorial Interaction & Morphology** | Disentangle community $\times$ epoch interaction effects and archaic case marking. |
| **Level 4** | **Conversational Dialogue Understanding** | Resolve pronominal anaphora, ellipsis (omitted verbs), and conversational repairs. |
| **Level 5** | **Counterfactual & OOD Generalization** | Correctly interpret paired counterfactual dialogues and unseen compositional pairings. |

---

### 4. Counterfactual Dialogue Testing

A key benchmark feature is **Counterfactual Context Pairs**:  
The contestant receives two episodes with the **exact same query utterance** $U_q$, but differing dialogue histories:

- **Episode 42A**:
  - Context: Speaker A says *"Scout Alpha will inspect the crystal core."*
  - Query: *"el id vidil."*
  - Correct Frame: `agent = "scout_alpha"`, `object = "crystal_core"`
- **Episode 42B**:
  - Context: Speaker A says *"Guardian Theta will inspect the ancient relic."*
  - Query: *"el id vidil."*
  - Correct Frame: `agent = "guardian_theta"`, `object = "ancient_relic"`

Models that rely on memorization or static priors will produce the same prediction for both episodes, scoring 0% on counterfactual consistency. Only systems that dynamically ground pronouns in conversation history succeed.

---

### 5. Input and Output Specification

#### Input Format (`episode` JSON):
```json
{
  "episode_id": "ep_0105",
  "community_id": "comm_north",
  "period_id": "period_archaic",
  "word_order": "SOV",
  "difficulty_level": 4,
  "support_set": [
    {
      "example_id": 1,
      "utterance": "thoral glacom vidost.",
      "event_frame": {
        "agent": "scout_alpha",
        "action": "inspect",
        "object": "crystal_core",
        "tense": "past",
        "polarity": "affirmative",
        "resolved_references": {}
      }
    }
  ],
  "query": {
    "context": [
      {
        "turn_id": 1,
        "speaker": "Speaker A",
        "utterance": "thorbe glacom vidost.",
        "event_frame": {
          "agent": "scout_beta",
          "action": "inspect",
          "object": "crystal_core",
          "tense": "past",
          "polarity": "affirmative"
        }
      }
    ],
    "utterance": "la glacom firmesc."
  }
}
```

#### Required Output Format:
```json
{
  "agent": "scout_beta",
  "action": "secure",
  "object": "crystal_core",
  "tense": "future",
  "polarity": "affirmative",
  "resolved_references": {
    "la": "scout_beta"
  },
  "drift_attribution": {
    "agent": "stable",
    "action": "interaction_shift",
    "object": "community_shift",
    "tense": "historical_shift",
    "polarity": "stable"
  }
}
```

---

### 6. Official Scoring Formula

Contestant submissions are evaluated on a hidden test set of 100 episodes:

$$\text{Final Score} = 0.40 \times \text{Exact Frame Acc} + 0.25 \times \text{Mean Slot F1} + 0.15 \times \text{Disentanglement Acc} + 0.10 \times \text{Reference Acc} + 0.10 \times \text{Counterfactual Consistency}$$

Partial credit is awarded at each level, ensuring novice students can earn meaningful points on Levels 1–2 while advanced competitors are challenged by Levels 4–5.
