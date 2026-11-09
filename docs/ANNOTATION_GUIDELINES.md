# Project SHIFT: Task Specification & Semantic Ontology Guidelines
## Milestone 02 Deliverable for IOAI 2027 Task Development

### 1. Event Frame Schema Specification

Every conversational event in Project SHIFT is mapped to an `EventFrame` instance conforming to the following JSON schema:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "SHIFT_EventFrame",
  "type": "object",
  "required": ["agent", "action", "object", "tense", "polarity", "resolved_references"],
  "properties": {
    "agent": {
      "type": "string",
      "enum": [
        "scout_alpha", "scout_beta",
        "artisan_gamma", "artisan_delta",
        "scholar_omega", "scholar_kappa",
        "guardian_theta", "guardian_sigma",
        "speaker", "addressee"
      ]
    },
    "action": {
      "type": "string",
      "enum": [
        "inspect", "harvest", "forge", "secure",
        "relocate", "repair", "activate", "abandon"
      ]
    },
    "object": {
      "type": "string",
      "enum": [
        "crystal_core", "ancient_relic", "signal_beacon", "water_purifier",
        "stone_monolith", "energy_matrix", "data_capsule", "shelter_dome"
      ]
    },
    "tense": {
      "type": "string",
      "enum": ["past", "present", "future"]
    },
    "polarity": {
      "type": "string",
      "enum": ["affirmative", "negative"]
    },
    "resolved_references": {
      "type": "object",
      "additionalProperties": { "type": "string" }
    }
  },
  "additionalProperties": false
}
```

---

### 2. Entity Definitions and Attributes

#### 2.1 Agents
| Entity ID | Role | Semantic Archetype | Gender / Agreement | Default Surface Root |
| :--- | :--- | :--- | :--- | :--- |
| `scout_alpha` | `scout` | Field surveyor & gatherer | Masculine (`el`) | `tor-al` |
| `scout_beta` | `scout` | Field surveyor & gatherer | Feminine (`la`) | `tor-be` |
| `artisan_gamma` | `artisan` | Crafter, engineer & builder | Masculine (`el`) | `kel-ga` |
| `artisan_delta` | `artisan` | Crafter, engineer & builder | Feminine (`la`) | `kel-de` |
| `scholar_omega` | `scholar` | Analyst, researcher & archivist | Masculine (`el`) | `mor-om` |
| `scholar_kappa` | `scholar` | Analyst, researcher & archivist | Feminine (`la`) | `mor-ka` |
| `guardian_theta` | `guardian` | Protector, sentry & security | Masculine (`el`) | `zar-th` |
| `guardian_sigma` | `guardian` | Protector, sentry & security | Feminine (`la`) | `zar-si` |
| `speaker` | `participant` | 1st person dialogue speaker | 1st person (`mi`) | `mi` |
| `addressee` | `participant` | 2nd person dialogue addressee | 2nd person (`tu`) | `tu` |

#### 2.2 Action Predicates
| Predicate | Transitivity | Semantic Definition | Selectional Affinity | Default Root |
| :--- | :--- | :--- | :--- | :--- |
| `inspect` | Transitive | Visual or sensor examination | Universal | `vid` |
| `harvest` | Transitive | Extracting or collecting resources | `crystal_core`, `water_purifier` | `fen` |
| `forge` | Transitive | Creating or assembling equipment | `signal_beacon`, `energy_matrix` | `vok` |
| `secure` | Transitive | Establishing security or fortification | `shelter_dome`, `stone_monolith` | `rak` |
| `relocate` | Transitive | Transporting to another site | Portable entities (`data_capsule`) | `lum` |
| `repair` | Transitive | Restoring damaged equipment | Technical infrastructure | `san` |
| `activate` | Transitive | Powering on or triggering system | Electronics & power systems | `dyn` |
| `abandon` | Transitive | Leaving behind or vacating | Fixed monuments & structures | `nul` |

#### 2.3 Objects
| Object ID | Category | Portability | Default Root |
| :--- | :--- | :--- | :--- |
| `crystal_core` | Energy | Portable | `kris` |
| `ancient_relic` | Artifact | Portable | `ark` |
| `signal_beacon` | Communications | Fixed | `bek` |
| `water_purifier` | Utility | Fixed | `hyd` |
| `stone_monolith` | Monument | Fixed | `mon` |
| `energy_matrix` | Infrastructure | Fixed | `mat` |
| `data_capsule` | Knowledge Archive | Portable | `dox` |
| `shelter_dome` | Habitat | Fixed | `dom` |

---

### 3. Reference Resolution Rules

1. **Pronominal Anaphora**:
   - `el` (masculine 3rd person) resolves to the most recently mentioned masculine agent in discourse context.
   - `la` (feminine 3rd person) resolves to the most recently mentioned feminine agent in discourse context.
   - `id` (neuter 3rd person) resolves to the most recently mentioned object in discourse context.
   - `mi` (1st person) resolves to `speaker`.
   - `tu` (2nd person) resolves to `addressee`.
2. **Ellipsis**:
   - When a sentence lacks a verb predicate (indicated syntactically by conjunction `et`), the action and tense are inherited from the salient event in the immediately preceding conversational turn.
   - The resolved references dictionary must record `{"[elliptic_verb]": "<canonical_action>"}`.
3. **Conversational Repair / Correction**:
   - When a turn begins with `sed` (*"but / rather"*), it updates the discourse state, overriding the antecedent proposition. Subsequent anaphoric references resolve to the corrected entity.
