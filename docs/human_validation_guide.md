# Project SHIFT: Human Validation & Cognitive Verification Guide
## Protocol for Independent Human Reviewers & IOAI Jury Members
**Target**: IOAI 2027 Singapore Benchmark Quality Assurance  
**Version**: 1.0 (Final Official Guidelines)

---

### 1. Purpose & Standards of Human Solvability

The goal of this guide is to ensure that **every episode in Project SHIFT is 100% human-interpretable and solvable from first principles** by a student or judge using only paper, pencil, and standard deduction within **5 to 8 minutes**.

#### Core Verification Criteria:
1. **Deductive Completeness**: Every morpheme, grammatical rule, and pronoun antecedent necessary to interpret the query must be explicitly present in the support set or preceding dialogue context.
2. **Deterministic Uniqueness**: There must exist exactly ONE valid parsed Event Frame. Ambiguous interpretations are considered benchmark errors and rejected.
3. **Zero Answer Leakage**: Reviewers solve the episode using the contestant-facing view (`to_contestant_dict()`), which contains NO hidden labels in either the query or preceding context turns.
4. **Inter-Annotator Agreement**: Independent human reviewers must achieve a Cohen's Kappa / Fleiss' Kappa of $\kappa \ge 0.95$ (near-perfect agreement).

---

### 2. Cognitive Deduction Procedure for Human Reviewers

A reviewer or contestant follows a structured 4-step deduction process:

```
[Support Examples] --------> Step 1: Lexicon Induction (Root Nouns & Verbs)
        |
        +------------------> Step 2: Morphosyntax (Word Order, Tense, Polarity, Case)
        |
[Dialogue Context] --------> Step 3: Discourse Tracking (Antecedents, Ellipsis, Repairs)
        |
        v
[Query Utterance] ---------> Step 4: Final Event Frame Synthesis
```

---

### 3. Step-by-Step Walkthroughs Across Difficulty Levels

#### Case Study A: Level 1 (Direct Lexicon Induction, SVO Word Order)
**Episode**: `level1_sample_01`  
**Metadata**: Community: `Coastal Valley`, Historical Period: `Classical`, Word Order: `SVO`

**Support Set (Excerpt)**:
1. `"zorin corin mon."` $\rightarrow$ `{agent: "artisan_delta", action: "harvest", object: "stone_monolith", tense: "present", polarity: "affirmative"}`
2. `"zorin dynin mon."` $\rightarrow$ `{agent: "artisan_delta", action: "activate", object: "stone_monolith", tense: "present", polarity: "affirmative"}`
3. `"celde corin hyd."` $\rightarrow$ `{agent: "scout_alpha", action: "harvest", object: "water_purifier", tense: "present", polarity: "affirmative"}`

**Query to Solve**:
- Preceding Context: None ($c=0$)
- Query Utterance: `"celde dynin hyd."`

**Human Cognitive Steps**:
1. **Lexical Alignment**:
   - Comparing Ex 1 and Ex 2: Subject is `"zorin"` (`artisan_delta`), Object is `"mon"` (`stone_monolith`).
   - Comparing Ex 1 and Ex 3: Verb is `"corin"` (`harvest`), Object is `"hyd"` (`water_purifier`). Subject in Ex 3 is `"celde"` (`scout_alpha`).
   - In Ex 2, verb is `"dynin"` (`activate`).
2. **Morphosyntax**:
   - Word order is `SVO`: `"celde"` (S) + `"dynin"` (V) + `"hyd"` (O).
   - Verb suffix `"-in"` maps to `tense: "present"`.
   - No negative prefix (`ne-`) or particle (`non`), so `polarity: "affirmative"`.
3. **Discourse Tracking**: N/A (no pronouns).
4. **Final Frame**:
   - `agent = "scout_alpha"`
   - `action = "activate"`
   - `object = "water_purifier"`
   - `tense = "present"`, `polarity = "affirmative"`
   - $\implies$ **100% Exact Match**. Deduction time: ~2 minutes.

---

#### Case Study B: Level 4 (Multi-Turn Dialogue Context with Pronominal Anaphora)
**Episode**: `level4_sample_04`  
**Metadata**: Community: `Northern Highlands`, Historical Period: `Contemporary`, Word Order: `SOV`

**Support Set (Excerpt)**:
1. `"thoral dom vidil."` $\rightarrow$ `{agent: "guardian_sigma", action: "inspect", object: "shelter_dome", tense: "past", polarity: "affirmative"}`
2. `"morom mon dynen."` $\rightarrow$ `{agent: "scholar_omega", action: "activate", object: "stone_monolith", tense: "present", polarity: "affirmative"}`
3. `"thoral mon vidil."` $\rightarrow$ `{agent: "guardian_sigma", action: "inspect", object: "stone_monolith", tense: "past", polarity: "affirmative"}`

**Dialogue Context**:
- Turn 1 (Speaker A): `"thoral dom vidil."`
  *(Note: Contestant view contains only this raw text string. The human parses Turn 1 using the support set)*:
  - `"thoral"` = `guardian_sigma`, `"dom"` = `shelter_dome`, `"vidil"` = `inspect` (past).
  - Discourse Salience: Primary Agent = `guardian_sigma`, Primary Patient = `shelter_dome`.

**Query Utterance (Speaker B)**:
- `"la id dynen."`

**Human Cognitive Steps**:
1. **Pronoun Recognition**:
   - Query contains pronouns `"la"` and `"id"` instead of overt noun stems.
   - Based on standard pronoun rules: `"la"` = 3rd-person feminine agreement pronoun; `"id"` = 3rd-person inanimate object pronoun.
2. **Discourse Antecedent Grounding**:
   - Looking at Turn 1:
     - The only introduced agent is `guardian_sigma` (feminine agreement matches `"la"`).
     - The only introduced patient object is `shelter_dome` (matches `"id"`).
   - Resolved References: `{"la": "guardian_sigma", "id": "shelter_dome"}`.
3. **Verb Induction**:
   - `"dynen"` matches Example 2: root `"dyn"` (`activate`) + suffix `"-en"` (`present`).
4. **Final Frame**:
   - `agent = "guardian_sigma"`
   - `action = "activate"`
   - `object = "shelter_dome"`
   - `tense = "present"`, `polarity = "affirmative"`
   - `resolved_references = {"la": "guardian_sigma", "id": "shelter_dome"}`
   - $\implies$ **100% Exact Match**. Deduction time: ~4 minutes.

---

#### Case Study C: Level 5 (Paired Counterfactual Dialogues)
**Twin Pair**: `cf_pair_12A` vs `cf_pair_12B`  
**Identical Query**: `"el id vidil."` (SOV, past tense)

**Context for 12A**:
- Turn 1: `"celde mon dynil."` $\rightarrow$ Introduces `scout_alpha` (`"celde"`) and `stone_monolith` (`"mon"`).
- Query Resolution for 12A: `agent = "scout_alpha"`, `object = "stone_monolith"`.

**Context for 12B**:
- Turn 1: `"zarsi dom dynil."` $\rightarrow$ Introduces `guardian_theta` (`"zarsi"`) and `shelter_dome` (`"dom"`).
- Query Resolution for 12B: `agent = "guardian_theta"`, `object = "shelter_dome"`.

**Human Verification**:
- The human reviewer checks that BOTH branches are unambiguously solvable and that the differing context forces a corresponding change in the query event frame.
- Verifies that `gt(A) != gt(B)` and both are uniquely grounded. Deduction time: ~5 minutes.

---

### 4. Inter-Annotator Agreement Protocol

To certify benchmark quality for IOAI 2027:
1. **Sample Selection**: 50 episodes sampled uniformly across all 5 difficulty levels.
2. **Dual Blind Annotation**: Two independent linguists/reviewers annotate the sampled episodes without seeing the ground-truth answer keys or each other's work.
3. **Agreement Metric**:
   - **Cohen's Kappa ($\kappa$)** calculated across all 5 frame slots (agent, action, object, tense, polarity):
     $$\kappa = \frac{P_o - P_e}{1 - P_e}$$
   - **Benchmark Certification Threshold**: $\kappa \ge 0.95$.
4. **Discrepancy Adjudication**: Any disagreement triggers immediate algorithmic review of the generator grammar to remove any potential lexical or structural ambiguity.
