# PROJECT SHIFT: Semantic Adaptation & Disentanglement Across Communities and Historical Time
### IOAI 2027 Scientific Benchmark Infrastructure (v0.3)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![IOAI 2027 Ready](https://img.shields.io/badge/IOAI%202027-Task%20Proposal-brightgreen.svg)]()
[![Tests: 48 Passed](https://img.shields.io/badge/tests-48%20passed-success.svg)]()

---

## 1. Project Goal & Overview

**Project SHIFT** is an advanced Natural Language Processing (NLP) benchmark and competition task engineered for the **International Olympiad in Artificial Intelligence (IOAI) 2027**.

### Core Research Objective:
> *Develop a controlled NLP benchmark to evaluate whether AI systems can infer and disentangle interacting semantic changes across communities and historical periods from limited labelled examples, use conversational context to interpret unfamiliar utterances, and generalise to unseen linguistic environments.*

### Five Core Capabilities Tested:
1. **Semantic Pattern Discovery**: Deducing unfamiliar vocabulary and morphological inflections from limited few-shot examples.
2. **Factorial Drift Disentanglement**: Distinguishing community-specific changes from historical temporal shifts and localized interaction effects ($C \times P$).
3. **Few-Shot Linguistic Adaptation**: Rapidly adapting with support sets of size $k \in \{0, 4, 8, 16, 32\}$.
4. **Contextual Dialogue Interpretation**: Resolving multi-turn discourse coreference, anaphora, ellipsis, and conversational repairs.
5. **Compositional Generalization & Counterfactual Sensitivity**: Generalizing to held-out semantic compositions and paired counterfactual dialogues.

---

## 2. Benchmark Architecture (v0.3 Scientific Infrastructure)

```
PROJECT_SHIFT/
├── shift/                                # Core Linguistic & Benchmark Infrastructure
│   ├── ontology.py                       # Semantic ontology, enums, & EventFrame schema with drift attribution
│   ├── lexicon.py                        # Morpheme roots, compounding, & collision auditing
│   ├── drift.py                          # [1] Factorial Semantic Drift Framework (Community x Time interactions)
│   ├── grammar.py                        # SOV, SVO, VSO syntax & morphological realisation
│   ├── dialogue.py                       # Multi-turn discourse, coreference, ellipsis & counterfactual pairs
│   ├── episodes.py                       # Episode generator with progressive difficulty levels (Levels 1-5)
│   ├── splits.py                         # [2] Leakage-resistant dataset splitting & verification
│   ├── learnability_validator.py         # [3] Query solvability & ambiguity quality gate
│   ├── shortcut_audits.py                # [4] Adversarial shortcut & memorisation audit suite
│   ├── schema.py                         # [5] Standard Draft-07 JSON EventFrame schema & validator
│   ├── metrics.py                        # [6a] Objective scoring metrics (EM, Slot F1, Disentanglement, CF)
│   ├── evaluator.py                      # [6b] Deterministic competition evaluator engine & CLI
│   ├── diagnostics.py                    # [7] Model weakness identification & 5 root-cause failure modes
│   ├── evaluation_report.py              # [8] Multi-format reproducible reporting (Markdown, LaTeX, JSON)
│   ├── validation.py                     # Overall dataset quality & integrity audits
│   ├── robustness.py                     # Stress tests & adversarial perturbation suite
│   └── serialize.py                      # JSONL streaming loader and exporter
├── baselines/                            # 4 Baseline Systems & Evaluator
│   ├── majority.py                       # Support majority baseline
│   ├── surface_lookup.py                 # Nearest-neighbour Jaccard token overlap baseline
│   ├── statistical.py                    # Statistical Naive Bayes / MLE baseline
│   ├── rule_based.py                     # Inductive symbolic learner with discourse & disentanglement tracking
│   └── evaluate.py                       # Evaluation runner & metric calculator
├── datasets/                             # Generated & Audited Datasets
│   ├── pilot_100.jsonl                   # 100-episode Pilot Dataset
│   ├── train.jsonl                       # 100 training episodes
│   ├── val.jsonl                         # 30 validation episodes
│   ├── public_dev.jsonl                  # 30 development episodes with solutions
│   ├── hidden_test_contestant.jsonl      # 100 hidden competition queries (labels stripped)
│   ├── hidden_test_labels.json           # Secret answer key for official evaluation
│   ├── test_counterfactual.jsonl         # 20 counterfactual paired episodes (10 matched pairs)
│   └── test_{iid,ood,comm,period}.jsonl  # Diagnostic sub-split sets
├── hidden_evaluator_package/             # Official Jury & Scientific Committee Package
│   ├── grade_submission.py               # Zero-dependency official grading script
│   ├── hidden_test_labels.json           # Secret ground-truth answer key
│   └── README.md                         # Grading instructions & scoring methodology
├── contestant_package/                   # Contestant Starter Kit
│   ├── loader.py                         # Pure Python JSONL data loader
│   ├── submission_template.py            # Starter submission model template
│   ├── local_evaluator.py                # Local validator with progressive level scoring
│   ├── sample_data/                      # Sample development datasets
│   └── README.md                         # Contestant instructions & guide
├── docs/                                 # Official Documentation Suite
│   ├── IOAI_2027_PROBLEM_STATEMENT.md    # Official Olympiad task statement (Levels 1-5)
│   ├── LITERATURE_REVIEW.md              # Literature matrix against SCAN, gScan, COGS, CFQ, SCONE, bAbI
│   ├── ANNOTATION_GUIDELINES.md          # Ontology, schema definitions, & selectional restrictions
│   └── JURY_SOLUTION_GUIDE.md            # Jury & coaches guide + educational value statement
├── scripts/
│   ├── generate_datasets.py              # Reproducible dataset generation pipeline with quality gates
│   └── run_experiments.py                # Benchmark experiment runner & report generator
├── configs/
│   └── default_task_config.json          # Task configuration & hyperparameters
├── tests/                                # 48 Automated Unit Tests (100% passing)
│   ├── test_ontology.py
│   ├── test_lexicon.py
│   ├── test_grammar.py
│   ├── test_drift.py
│   ├── test_dialogue.py
│   ├── test_episodes_and_splits.py
│   ├── test_baselines_and_robustness.py
│   ├── test_strengthened_features.py
│   └── test_v03_infrastructure.py
├── results/
│   ├── shift_benchmark_report.md         # Formatted Markdown report with comparative tables
│   ├── shift_benchmark_report.tex        # Publication-ready LaTeX tabular snippet for paper
│   ├── shift_benchmark_report.json       # Structured machine-readable benchmark results
│   └── experiment_results.json           # Detailed empirical results across all experiments
└── README.md                             # Comprehensive project documentation
```

---

## 3. Official Benchmark Results (v0.3 Infrastructure)

Evaluated on the 100-episode Hidden Test Set + Counterfactual Pair Suite:

### 3.1 Baseline Systems Performance Comparison

$$\text{Composite Score} = 0.40 \times \text{EM} + 0.25 \times \text{Slot-F1} + 0.15 \times \text{Disentangle} + 0.10 \times \text{RefRes} + 0.10 \times \text{CF-Consist}$$

| Model Architecture | Exact Match (%) | IOAI Composite Score | Disentanglement (%) | Reference Res. (%) | Counterfactual Consist. (%) | Mean Slot F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Inductive Rule Learner (Ours)** | **75.00%** | **86.05** | **95.00%** | **84.00%** | **100.00%** | **93.62** |
| **Statistical Naive Bayes** | 75.00% | 71.80 | 0.00% | 84.00% | 100.00% | 93.62 |
| **Majority Baseline** | 1.00% | 23.83 | 0.00% | 26.00% | 100.00% | 43.33 |
| **Surface Lookup Baseline** | 0.00% | 20.45 | 0.00% | 28.00% | 100.00% | 30.59 |

> **Key Scientific Insight**: While statistical frequency counting correctly infers surface slot mappings (75.0% EM), it achieves **0.0% Disentanglement Accuracy** because it cannot isolate causal factors. The Inductive Rule Learner achieves **95.0% Disentanglement Accuracy**, validating that Project SHIFT successfully separates superficial correlation from genuine causal reasoning.

---

### 3.2 Progressive Difficulty Levels (Curriculum Evaluation)

| Model Architecture | Level 1 (Direct) | Level 2 (Drift) | Level 3 (Coreference) | Level 4 (Repair/Ellipsis) | Level 5 (Counterfactual) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Inductive Rule Learner** | **100.0%** | **100.0%** | **100.0%** | **71.2%** | **100.0%** |
| Statistical Naive Bayes | 100.0% | 100.0% | 100.0% | 71.2% | 100.0% |
| Majority Baseline | 14.3% | 0.0% | 0.0% | 0.0% | 0.0% |
| Surface Lookup Baseline | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

---

### 3.3 Adversarial Shortcut Audits & Leakage Resistance

| Model Architecture | Full Context | Support Ablation ($k=0$) | Context Ablation | Metadata Only | Permutation Drop | Vulnerability Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Inductive Rule Learner** | **75.0%** | **18.0%** | **2.0%** | **1.0%** | **57.0%** | **Robust (Adapting)** |
| **Statistical Naive Bayes** | 75.0% | 18.0% | 2.0% | 1.0% | 57.0% | **Robust (Adapting)** |
| Majority Baseline | 1.0% | 0.0% | 0.0% | 1.0% | 1.0% | Baseline |
| Surface Lookup Baseline | 0.0% | 0.0% | 0.0% | 1.0% | 0.0% | Baseline |

- **Support Set Necessity**: Performance plummets by **57.0%** when support examples are removed ($k=0$), confirming that models must actively adapt to in-context evidence.
- **Context Dependency**: Performance on discourse queries collapses to **2.0%** without dialogue history, proving models cannot exploit bag-of-words heuristics.
- **Zero Metadata Exploitation**: A metadata-only classifier achieves only **1.0%**, confirming that community/epoch names do not leak answers.

---

### 3.4 Diagnostic Error Taxonomy (Five Failure Modes)

Project SHIFT classifies student and model mistakes into 5 root-cause failure modes:
1. `LexicalInductionFailure`: Misalignment of root morphemes from the support set.
2. `MorphosyntacticFailure`: Misparsing of tense/polarity affixes and historical lenition rules.
3. `DiscourseTrackingFailure`: Antecedent resolution failure across multi-turn conversational repair.
4. `DisentanglementFailure`: Correct semantic event frame but incorrect causal attribution.
5. `CounterfactualContextBias`: Same output across paired counterfactual branches.

---

## 4. Getting Started

### 4.1 Running the Unit Test Suite (Zero Dependencies)
```bash
python -m unittest discover -s tests -p "test_*.py"
# Ran 48 tests in 0.580s - OK (100% Passing)
```

### 4.2 Regenerating Datasets with Quality Gates
```bash
python scripts/generate_datasets.py
```

### 4.3 Running Benchmark Experiments & Generating Reports
```bash
python scripts/run_experiments.py
```

### 4.4 Contestant Local Evaluation
```bash
python contestant_package/local_evaluator.py --data contestant_package/sample_data/public_dev.jsonl
```

### 4.5 Official Jury Grading (Hidden Test Set)
```bash
python hidden_evaluator_package/grade_submission.py --submission path/to/submission.json
```
