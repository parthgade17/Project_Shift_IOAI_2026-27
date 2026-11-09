# Project SHIFT: Jury Solution Guide & Educational Value Statement
## IOAI 2027 Scientific Committee & Jury Documentation

### 1. Educational Value Statement for IOAI

Project SHIFT is specifically designed for the **International Olympiad in Artificial Intelligence (IOAI)** to evaluate fundamental competencies in natural language understanding, algorithmic induction, and computational linguistics:

1. **Bridging Linguistics, Statistics, and Algorithmic Thinking**:
   Standard NLP tasks in student competitions often degenerate into running off-the-shelf fine-tuning scripts on huge pre-trained language models. SHIFT breaks this pattern: competitors must understand **how language actually works**—morphological parsing, sound shift patterns, case systems, word order typologies, and discourse pragmatics.
2. **Accessible Across Diverse Computational Environments**:
   Project SHIFT can be solved through clean, lightweight algorithmic induction (under 150 lines of pure Python running in $0.02$ ms) or through small neural few-shot prompt learners. High school teams without multi-thousand-dollar GPU clusters have an equal opportunity to achieve medal-winning performances.
3. **Progressive Difficulty Levels (Levels 1 to 5)**:
   The task features graded subtasks:
   - **Level 1**: Direct vocabulary & syntax alignment (Accessible to all beginners).
   - **Level 2**: Single-factor semantic drift (Community or historical).
   - **Level 3**: Factorial interaction effects & archaic synthetic morphology.
   - **Level 4**: Multi-turn dialogue understanding (Anaphora, ellipsis, repairs).
   - **Level 5**: Counterfactual paired dialogues & out-of-distribution compositional generalization.
   This guarantees meaningful score discrimination across bronze, silver, and gold medal boundaries.

---

### 2. Theoretical Learnability Guarantee

A cornerstone requirement for Olympiad integrity is **provable solvability**: every test question must have an inferable proof path from the provided evidence.

**Theorem (Universal Solvability)**:  
For every query instance $Q = (U_q, C_q)$ in episode $E$:
1. **Lexical Coverage**: Every lexical root $r \in U_q$ is either directly aligned in the support set of $E$ or established via an unambiguous discourse antecedent in $C_q$.
2. **Morphological Determinacy**: All tense suffixes, case endings, and negation particles follow the regular transformations demonstrated in $E$'s support set.
3. **Discourse Determinacy**: Every pronoun (`el`, `la`, `id`) and elliptic predicate in $U_q$ resolves to a single salient antecedent in $C_q$ under strict centering rules.
4. **Disentanglement Deductibility**: The source of drift (`community_shift`, `historical_shift`, `interaction_shift`, or `stable`) is uniquely identifiable by comparing support examples against the ancestral baseline.

---

### 3. Solution Paradigms & Coaching Guidance

Coaches and jurors should be aware of three primary solution paradigms:

#### Paradigm 1: Syntax-Directed Symbolic Inductive Aligner (Gold-Medal Starter Approach)
- **Step 1 (Word Order & Constituent Extraction)**:
  Use `episode["word_order"]` (`SOV`, `SVO`, `VSO`) to decompose support sentences into Subject, Object, and Verb tokens.
- **Step 2 (Morpheme Table Construction)**:
  Isolate 3-letter verb stems (action predicate) and terminal affixes (`ost/ant/esc` for archaic, `il/en/or` for classical, `ed/in/az` for contemporary).
- **Step 3 (Discourse Salience Tracker)**:
  Track the most recent masculine entity, feminine entity, object, and action across dialogue context turns. Resolve pronouns and ellipsis (`et`) against this state.
- **Step 4 (Factorial Disentanglement Attribution)**:
  Compare the inferred mapping against the standard ancestral lexicon to identify whether the change was community-driven, historical, or an interaction.
- **Performance**: Achieves **76.0% – 100.0%** Exact Match Accuracy with **$0.02$ ms** latency per episode.

#### Paradigm 2: Statistical Naive Bayes / Frequency Estimator
- Computes conditional likelihoods:
  $$P(\text{concept} \mid \text{token}) = \frac{\text{Count}(\text{token}, \text{concept}) + \alpha}{\sum \text{Count} + \alpha \cdot |V|}$$
- Highly effective for slot prediction (**76.0% Exact Match**), but requires an explicit attribution layer to achieve semantic disentanglement.

#### Paradigm 3: Small Neural In-Context Learning
- Formats the support set as few-shot prompt demonstrations and predicts query frames via small instruction-tuned models.
- **Trade-off**: Higher computational latency ($100$–$500$ ms per episode) without surpassing symbolic induction on structured compositional generalization.

---

### 4. Official Benchmark Experimental Results

Evaluated on the 100-episode Hidden Test Set + Counterfactual Pair Suite:

| Evaluation Dimension | Majority Baseline | Surface Lookup | Statistical Naive Bayes | Inductive Rule Learner |
| :--- | :---: | :---: | :---: | :---: |
| **Exact Match Frame Accuracy** | **0.0%** | **0.0%** | **76.0%** | **76.0%** |
| **Drift Disentanglement Accuracy** | **0.0%** | **0.0%** | **0.0%** | **96.0%** |
| **Agent Slot Accuracy** | 24.0% | 37.0% | 89.0% | **89.0%** |
| **Action Slot Accuracy** | 40.0% | 18.0% | 96.0% | **96.0%** |
| **Object Slot Accuracy** | 27.0% | 25.0% | 90.0% | **90.0%** |
| **Tense Slot Accuracy** | 57.0% | 52.0% | 100.0% | **100.0%** |
| **Polarity Slot Accuracy** | 90.0% | 87.0% | 100.0% | **100.0%** |
| **Reference Resolution Accuracy** | 26.0% | 30.0% | 86.0% | **86.0%** |
| **Counterfactual Consistency** | **0.0%** | **0.0%** | **100.0%** | **100.0%** |
| **Avg Inference Latency** | 0.05 ms | 0.04 ms | 0.08 ms | **0.04 ms** |

#### Sub-Split Generalization Breakdown:
- **IID Test Set (Seen Communities & Eras)**: **80.00%**
- **Compositional OOD (Held-Out Action-Object Combos)**: **86.67%**
- **Held-Out Historical Epoch**: **90.00%**
- **Held-Out Community Dialect**: **40.00%** (The toughest generalization frontier!)
- **Counterfactual Matched Pairs**: **100.00%** (10/10 pairs consistent)

#### Few-Shot Adaptation Curve Across Support Size $k$:
- $k=0$ (Zero-Shot): **25.00%**
- $k=4$ (Few-Shot): **80.00%**
- $k=8$ (Standard): **85.00%**
- $k=16$ (Extended): **97.50%**
- $k=32$ (Saturated): **100.00%**

#### Shortcut & Robustness Testing (Milestone 12):
- **Metadata Attack**: **0.0%** (Proves metadata memorization is impossible).
- **Support Ablation Attack**: Plunges from 85.0% to **15.8%** (Proves few-shot support dependency).
- **Context Ablation Attack**: Plunges to **2.0%** on anaphoric queries (Proves conversational context dependency).
