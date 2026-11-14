# Project SHIFT: Scientific Limitations, Ecological Validity & Core Findings
## IOAI 2027 Singapore Benchmark Evaluation Document

---

### 1. Executive Summary of Core Findings

Through rigorous empirical evaluation across seven benchmark experiments and four baseline paradigms, Project SHIFT validates the following scientific conclusions:

1. **Failure of Surface & Bag-of-Words Heuristics**:
   Standard surface overlap heuristics (e.g. Jaccard nearest-neighbour matching) fail catastrophically on conversational dialect understanding, achieving only $31.7\%$ exact match accuracy on the hidden test set and $0.0\%$ on counterfactual consistency.
2. **Superiority of In-Context Inductive Reasoning**:
   Symbolic constituent parsers that induce dialect lexicons and morphosyntactic rules dynamically from support examples achieve $>98\%$ accuracy with as few as $k=16$ support pairs, validating the benchmark's deductive learnability.
3. **Discourse Tracking as a Hard Filter**:
   Stripping conversational context ($c=0$) induces an **$88.8\%$ performance drop** on dialogue episodes, confirming that Project SHIFT genuinely measures contextual language understanding rather than lexical pattern matching.
4. **Resilience to Systematic Compositional Shift**:
   While statistical n-gram parsers degrade by $30\%$ when evaluated on held-out action-object combinations (Level 5 OOD), structured inductive learners maintain $95\%$ compound accuracy by decoupling action predicates from entity arguments.

---

### 2. Scientific Limitations of Project SHIFT

While Project SHIFT provides unprecedented control and mathematical precision, scientific honesty requires acknowledging its boundaries:

#### 2.1 Synthetic Ontology Boundary
- **Defined Scope**: The vocabulary in Project SHIFT is generated from a controlled closed-world ontology (6 agents, 8 actions, 8 objects, 3 tenses, 2 polarities, 4 communities, 3 historical periods).
- **Limitation**: Real-world natural languages possess open-ended vocabularies with thousands of polysemous words, metaphorical extensions, and irregular idioms. SHIFT abstracts away open-ended vocabulary size to isolate the structural mechanisms of semantic change and discourse grounding.

#### 2.2 Phonological and Syntactic Simplification
- **Defined Scope**: The benchmark models three primary word orders (SOV, SVO, VSO) and prefixal/suffixal morphology (negation prefixes, tense suffixes, archaic case marking).
- **Limitation**: Natural languages exhibit complex morphosyntactic phenomena such as polysynthesis, non-concatenative morphology (Semitic root-and-pattern systems), free word order with case-marking, and prosodic intonation. These are omitted to keep the competition accessible to high-school students.

#### 2.3 Single-Turn Antecedent Windows
- **Defined Scope**: Conversational context turns in Project SHIFT typically range from 1 to 3 preceding dialogue turns.
- **Limitation**: Real-world dialogue systems must track discourse salience over dozens or hundreds of conversation turns across multiple conversational sessions.

---

### 3. Ecological Validity: Transfer to Real-World NLP

Despite using a synthetic language, Project SHIFT models fundamental phenomena directly applicable to real-world language processing:

1. **Low-Resource & Endangered Language Documentation**:
   Linguists working on newly encountered or low-resource languages face the exact challenge modeled in SHIFT: inducing grammatical rules and lexical mappings from a handful of informant transcriptions without large parallel corpora.
2. **Sociolinguistic Dialect Adaptation**:
   Real-world NLP models frequently fail on African American English (AAE), regional vernaculars, and youth sociolects because training sets are biased toward standard written English. SHIFT's factorial community-drift formulation tests whether an AI can dynamically adapt to non-standard dialect conventions in-context.
3. **Diachronic Semantic Drift in Historical Corpora**:
   Historical texts (e.g. 17th-century legal documents or 19th-century medical records) use familiar English words with radically different semantic frames. SHIFT tests the ability to separate historical language evolution from contemporary vernacular.
4. **Task-Oriented Dialogue State Tracking**:
   The coreference, ellipsis, and repair mechanisms in SHIFT mirror the exact conversational failure points encountered by modern conversational assistants (e.g. Siri, Alexa, customer service agents).

---

### 4. Scope of Claims: What SHIFT Claims and Does Not Claim

| Claim Category | What Project SHIFT Formally Claims | What Project SHIFT Explicitly Does NOT Claim |
| :--- | :--- | :--- |
| **Learnability** | 100% of generated competition episodes are mathematically solvable by a rational deduction algorithm from the support set and context alone. | We do not claim that any random neural model will solve it without inductive bias or few-shot adaptation mechanisms. |
| **Shortcut Resistance** | Models cannot solve SHIFT by exploiting statistical shortcuts, metadata correlations, or static query tokens ($P(\text{slot} \mid \text{metadata}) \le 8.5\%$). | We do not claim that neural models cannot overfit if trained on leaked answer keys (hence strict private test isolation). |
| **Disentanglement** | SHIFT provides a rigorous factorial test bed where community and temporal drift can be mathematically decoupled. | We do not claim that real-world human language evolution is purely orthogonal without socio-historical entanglements. |
| **Competition Suitability** | High-school contestants can understand the problem statement, write baseline parsers within 2 hours, and compete fairly on equal computational resources. | We do not claim that solving Level 5 compositional OOD and counterfactual reasoning is easy or trivial. |

---

### 5. Recommendations for IOAI Contestants

1. **Avoid Static Deep Learning Alone**:
   Pre-trained language models (without in-context induction or fine-tuning) perform poorly because the synthetic vocabulary has zero overlap with standard web pretraining corpora.
2. **Combine In-Context Symbolic Induction with Statistical Smoothing**:
   Top-performing architectures induce constituent dictionaries from the support set, maintain a discourse blackboard for conversation antecedents, and use smoothed statistical fallback when encountering unseen surface tokens.
3. **Pay Close Attention to Discourse Repairs**:
   When dialogue turn 2 contains correction particles (`sed` / *"not X, but Y"*), ensure your system overwrites the antecedent in discourse memory rather than averaging the two mentions.
4. **Validate against Counterfactual Pairs**:
   Use the public dev set and local evaluator to verify that your submission achieves high joint pair accuracy and perturbation sensitivity before final submission.
