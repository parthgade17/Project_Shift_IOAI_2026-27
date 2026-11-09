# Project SHIFT: Benchmark Scope, Scientific Validity & Limitations
## IOAI 2027 Task Scope and Governance

### 1. What Project SHIFT Measures

Project SHIFT is specifically designed to measure five foundational capabilities in artificial intelligence:
1. **In-Context Lexicon & Grammar Induction**: The capacity of an algorithm or model to infer the syntactic order (`SOV`, `SVO`, `VSO`), morpheme-to-concept bindings, and inflectional affixes (tense suffixes, polarity particles) from a small support set ($k \le 16$) without parameter updates.
2. **Causal Drift Disentanglement**: The ability to isolate the causal source of semantic divergence—specifically separating community-specific lexical conventions from historical language-wide shifts and localized interaction effects ($C \times P$).
3. **Discourse-State Anaphora Tracking**: The capacity to maintain entity salience across multi-turn conversational interactions, correctly resolving anaphoric pronouns (`la`, `id`, `te`), verb ellipsis (`[elliptic_verb]`), and conversational corrections across distinct speakers.
4. **Compositional Generalization (Systematicity)**: The ability to interpret novel combinations of familiar actions and objects that never co-occurred in the training distribution.
5. **Counterfactual Context Sensitivity**: The robustness of an agent to produce diverging interpretations when identical surface queries appear in contrasting conversational discourse histories.

---

### 2. What Project SHIFT Does NOT Measure

To maintain rigorous scientific clarity, we explicitly define the boundaries of Project SHIFT:
1. **NOT a Test of Real-World Historical Linguistics or Philology**:
   - SHIFT does not evaluate knowledge of Latin, Old English, Sanskrit, or real historical language families.
   - Success in SHIFT demonstrates algorithmic and mathematical capacity for semantic induction, not real-world philological expertise.
2. **NOT a Test of Real-World Sociolinguistic Nuance**:
   - Human sociolinguistics involves complex social hierarchies, power dynamics, pragmatics, and identity signaling.
   - SHIFT's "communities" and "historical eras" are mathematically controlled discrete variables in a factorial generative framework ($C \times P$), not sociological simulations.
3. **NOT a Test of World Knowledge or Commonsense Reasoning**:
   - The semantic ontology is closed ($|\text{Agents}|=6$, $|\text{Actions}|=6$, $|\text{Objects}|=6$).
   - The benchmark intentionally avoids commonsense priors so that models cannot cheat using pre-existing training corpora knowledge.

---

### 3. Scientific Justification for Synthetic Language Design

| Scientific Criterion | Natural Corpus Approach (e.g. Historical Twitter/Newspapers) | Project SHIFT Synthetic Framework |
| :--- | :--- | :--- |
| **Ground-Truth Determinacy** | Low. Natural language semantic drift is ambiguous, graded, and subjective across annotators. | **100% Deterministic**. Every semantic mapping is mathematically defined by the generative matrix. |
| **Confounding Variables** | High. Vocabulary shifts correlate with technology, politics, genre, and topic changes. | **Zero Confounding**. Topic, genre, and length are controlled orthogonally across communities and eras. |
| **Leakage & Memorization** | Severe. Public LLMs have ingested all public historical text and internet slang. | **Zero Leakage**. Synthetic roots and grammar rules cannot be memorized from internet pretraining data. |
| **Learnability Proof Path** | Unknown. Unsupervised corpus queries often lack necessary context for deduction. | **Guaranteed Solvability**. `LearnabilityValidator` proves that every query is solvable from the support set. |

---

### 4. Known Limitations & Boundary Conditions

1. **Discrete vs. Continuous Drift**:
   - *Limitation*: SHIFT models semantic drift across discrete historical epochs (Archaic, Classical, Modern) and discrete communities rather than continuous time intervals.
   - *Justification*: Discrete factorial modeling enables exact mathematical disentanglement scoring and deterministic evaluation suitable for competitive Olympiad environments.
2. **Closed Ontology Scale**:
   - *Limitation*: The domain consists of 6 canonical agents, 6 canonical actions, and 6 canonical objects (36 base event compositions).
   - *Justification*: High-stakes competition tasks require deterministic, human-auditable ground truths where every single test query can be manually verified by the jury in case of contestant appeals.
3. **Support Set Horizon**:
   - *Limitation*: Support sets are bounded at $k \in \{0, 4, 8, 16, 32\}$.
   - *Justification*: This exactly mirrors the computational constraints of rapid in-context adaptation in competitive AI challenges.

---

### 5. Ethical Guidelines & Responsible Research Design

1. **No Stereotyping or Real-Group Mimicry**:
   - All community names and languages are fictionalized, abstract synthetic names (e.g., *Citadel*, *Northern Highlands*, *Coastal Valley*).
   - No synthetic language borrows phonology or grammar to parody or mock real indigenous, regional, or endangered human languages.
2. **Equal Compute Access**:
   - Evaluation models require zero GPU hardware or paid cloud API tokens. The entire benchmark suite and baseline models run on standard consumer laptop CPUs in under 5 minutes.
