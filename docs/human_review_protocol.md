# Project SHIFT: Human Review & Verification Protocol
## IOAI 2027 Independent Jury Quality Assurance

This protocol defines the exact procedure for independent human verification of Project SHIFT episodes, guaranteeing human interpretability, instructional clarity, and annotation correctness before contest deployment.

---

### 1. Human Solvability & Interpretability Standard

For an episode to be certified as competition-ready:
1. **Deductive Completeness**: A high-school student or judge with no prior exposure to the synthetic language must be able to deduce the exact event frame using pen and paper within **5 to 8 minutes**.
2. **Deterministic Solution**: Two independent human reviewers must arrive at the exact same parsed event frame ($100\%$ inter-annotator agreement).
3. **No Guessing**: Reviewers must not need to guess any root or affix; all evidence must be explicit in the support set or preceding conversational turns.

---

### 2. Step-by-Step Human Reviewer Walkthrough

Reviewers follow this 4-step deduction procedure:

#### Sample Episode: `dev_0002` (Northern Highlands, Classical Epoch, SOV syntax)

**Support Set Provided (excerpt):**
- Example 1: `"celde mon dynil."` $\rightarrow$ `{agent: "artisan_delta", action: "activate", object: "stone_monolith", tense: "past", polarity: "affirmative"}`
- Example 2: `"zarsi dom dynil."` $\rightarrow$ `{agent: "guardian_sigma", action: "activate", object: "shelter_dome", tense: "past", polarity: "affirmative"}`
- Example 3: `"morom hyd coren."` $\rightarrow$ `{agent: "scholar_omega", action: "harvest", object: "water_purifier", tense: "present", polarity: "affirmative"}`
- Example 4: `"thoral mon nesanen."` $\rightarrow$ `{agent: "scout_alpha", action: "repair", object: "stone_monolith", tense: "present", polarity: "negative"}`

**Query to Solve:**
- Conversational Context Turn 1 (Speaker A): `"zarsi mon coril."` $\rightarrow$ `{agent: "guardian_sigma", action: "harvest", object: "stone_monolith", tense: "past", polarity: "affirmative"}`
- Query Utterance (Speaker B): `"la id dynil."`

#### Step 1: Lexical Morpheme Induction
1. Comparing Example 1 and Example 2:
   - Both share action `"activate"`, tense `"past"`, and token `"dynil"`. Therefore, root verb `"dyn"` = `"activate"`, and suffix `"-il"` = `"past"`.
2. In Example 1, `"mon"` maps to object `"stone_monolith"`.
3. In Example 2, `"dom"` maps to object `"shelter_dome"`.
4. Therefore, `"celde"` = `"artisan_delta"`, and `"zarsi"` = `"guardian_sigma"`.

#### Step 2: Morphosyntax Deduction
- Word order is `SOV`: Subject (`celde`) + Object (`mon`) + Verb (`dynil`).
- Negation prefix check: Example 4 has `"nesanen"` with label `polarity: "negative"` and action `"repair"`. Suffix `"-en"` = `"present"`. Prefix `"ne-"` = negation.
- Query utterance has no `"ne-"`, so `polarity = "affirmative"`.

#### Step 3: Discourse & Anaphora Tracking
- Query utterance is `"la id dynil"`.
- It contains pronouns `"la"` and `"id"`.
- Look at Turn 1: Speaker A introduced Subject `"zarsi"` (`"guardian_sigma"`) and Object `"mon"` (`"stone_monolith"`).
- By anaphora resolution rules:
  - `"la"` refers to the primary discourse agent antecedent: `"guardian_sigma"`.
  - `"id"` refers to the primary discourse object antecedent: `"stone_monolith"`.

#### Step 4: Final Deductive Synthesis
- **Agent**: `"guardian_sigma"` (via `"la"`)
- **Action**: `"activate"` (via `"dyn"`)
- **Object**: `"stone_monolith"` (via `"id"`)
- **Tense**: `"past"` (via `"-il"`)
- **Polarity**: `"affirmative"`
- **Resolved References**: `{"la": "guardian_sigma", "id": "stone_monolith"}`

**Result**: 100% exact match with ground-truth label. Proof path is sound, deterministic, and free of ambiguity.

---

### 3. Human Review Checklist for Contest Datasets

Every generated dataset undergoes this five-point audit by two independent reviewers:
- [x] **Instruction Clarity**: Task instructions define all 6 agents, 6 actions, 6 objects, 3 tenses, 2 polarities, and pronoun symbols clearly.
- [x] **Support Sufficiency**: Every root noun, verb, and tense suffix in the query appears at least once in the support set.
- [x] **Unambiguous Antecedents**: In multi-turn dialogue, every pronoun has exactly one salient antecedent in the immediate conversational history.
- [x] **Fairness of Repairs**: If Turn 2 corrects Turn 1 (e.g. `"sed..."`), the corrected entity completely supersedes the prior entity.
- [x] **Zero Reliance on Natural Language Assumptions**: No knowledge of English semantics is required beyond reading the support labels.
