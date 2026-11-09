# Project SHIFT: Dataset & Benchmark Card
## Formal Dataset Card for IOAI 2027 Scientific Benchmark

Following the standard specifications of *Datasheets for Datasets* (Gebru et al., 2021) and Hugging Face Dataset Cards.

---

### 1. Dataset Summary

- **Benchmark Name**: Project SHIFT (Semantic Adaptation & Disentanglement Across Communities and Historical Time)
- **Version**: `v0.3.0` (Production Olympiad Infrastructure)
- **Primary Task**: Few-Shot In-Context Semantic Parsing & Causal Drift Disentanglement
- **Modality**: Synthetic Multi-Turn Conversational Dialogue & Event Frames
- **Language**: Controlled Synthetic Language Families (Citadel, Northern Highlands, Coastal Valley across Archaic, Classical, and Modern Epochs)
- **Target Competition**: International Olympiad in Artificial Intelligence (IOAI) 2027

---

### 2. Dataset Structure

#### 2.1 Partitions & Episode Counts

| Split Name | Episode Count | Purpose | Label Availability |
| :--- | :---: | :--- | :---: |
| `train.jsonl` | 100 | In-distribution algorithmic development and few-shot tuning | Public |
| `val.jsonl` | 30 | Validation hyperparameter selection and syntax tuning | Public |
| `public_dev.jsonl` | 30 | Benchmark sanity checks, local evaluator testing, and error diagnosis | Public |
| `hidden_test_contestant.jsonl` | 100 | Private competitive evaluation for IOAI jury | Private (Labels stripped) |
| `hidden_test_labels.json` | 100 | Secret official ground-truth answer key held by jury | Secret |
| `test_counterfactual.jsonl` | 20 | 10 matched counterfactual pairs testing context sensitivity | Evaluation |
| `test_iid.jsonl` | 30 | Diagnostic in-distribution test split | Evaluation |
| `test_ood.jsonl` | 30 | Compositional out-of-distribution (unseen action-object combos) | Evaluation |
| `test_heldout_community.jsonl` | 20 | Unseen dialect community adaptation | Evaluation |
| `test_heldout_period.jsonl` | 20 | Unseen historical epoch adaptation | Evaluation |

#### 2.2 Schema Definition (Draft-07 JSON)
Each episode contains:
- `episode_id` (`str`): Unique identifier (e.g., `test_iid_0001`).
- `community_id` (`str`): Sociolinguistic group (`comm_citadel`, `comm_north`, `comm_coast`).
- `period_id` (`str`): Historical epoch (`period_archaic`, `period_classical`, `period_modern`).
- `word_order` (`str`): Syntactic structure (`SOV`, `SVO`, `VSO`).
- `support_set` (`List[SupportExample]`): $k$ labelled pairs of `(utterance, event_frame)`.
- `query` (`QueryInstance`): Unlabelled query with `dialogue` history and `phenomenon`.
- `difficulty_level` (`int`): Progressive complexity level ($1$ to $5$).
- `query_label` (`EventFrame`): Ground-truth target frame (in reference splits).

---

### 3. Dataset Curation & Quality Gates

1. **Orthogonal Factorial Generation**: Community and epoch transformations are independently parameterized in `shift/drift.py`.
2. **Automated Quality Gates**:
   - `SchemaValidator`: Enforces enum validity, non-empty fields, and valid reference dictionaries.
   - `LearnabilityValidator`: Mathematical proof that all lexical roots and affixes in the query have grounding in the support set.
   - `ShortcutAuditor`: Validates that metadata-only and $k=0$ attacks fail.
3. **Cross-Split Leakage Protection**: Verified with 0% query utterance or event-frame leakage across splits.

---

### 4. Intended Uses & Limitations

- **Intended Use**: Evaluating in-context semantic parsing, rapid few-shot adaptation, multi-turn conversational coreference, and causal disentanglement in algorithmic systems and LLMs.
- **Out-of-Scope Use**: Should NOT be used to draw empirical conclusions about real-world human history, sociolinguistics, or natural language etymology.

---

### 5. Maintenance & Reproducibility
- **Lead Maintainer**: Project SHIFT Research & Benchmark Committee
- **Licensing**: MIT Open Source License
- **Environment**: Pure Python 3.10+ standard library (Zero external network or GPU requirements).
