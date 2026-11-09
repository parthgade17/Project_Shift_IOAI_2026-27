# Project SHIFT: Official Benchmark Results (IOAI 2027)

**Benchmark Version**: Project SHIFT v0.3 (IOAI 2027 Scientific Benchmark Infrastructure)  
**Target Domain**: In-Context Semantic Adaptation & Drift Disentanglement  

---

## 1. Overall Model Performance Comparison

The table below compares baseline models on exact event-frame accuracy, the official IOAI composite score (partial credit), semantic drift disentanglement, reference resolution, and counterfactual consistency.

| Model Architecture | Exact Match (%) | IOAI Composite Score | Disentanglement (%) | Reference Res. (%) | Counterfactual Consist. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Inductive Rule Learner** | 75.00% | **86.05** | 95.00% | 84.00% | 100.00% |
| **Statistical Naive Bayes** | 75.00% | **71.80** | 0.00% | 84.00% | 100.00% |
| **Majority Baseline** | 1.00% | **23.83** | 0.00% | 26.00% | 100.00% |
| **Surface Lookup Baseline** | 0.00% | **20.45** | 0.00% | 28.00% | 100.00% |

## 2. Slot-Level Macro F1 Breakdown

Macro F1 scores across individual semantic frame slots:

| Model Architecture | Agent F1 | Action F1 | Object F1 | Tense F1 | Polarity F1 | Mean Slot F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Inductive Rule Learner | 88.26 | 93.88 | 85.94 | 100.00 | 100.00 | **93.62** |
| Statistical Naive Bayes | 88.26 | 93.88 | 85.94 | 100.00 | 100.00 | **93.62** |
| Majority Baseline | 19.53 | 37.38 | 29.17 | 59.67 | 70.91 | **43.33** |
| Surface Lookup Baseline | 34.77 | 16.07 | 12.04 | 41.87 | 48.19 | **30.59** |

## 3. Progressive Difficulty Levels (Curriculum Evaluation)

- **Level 1**: Direct Canonical Observation (Isolated context, fixed syntax)  
- **Level 2**: Unseen Semantic Drift (Factorial community/historical shift)  
- **Level 3**: Multi-Turn Dialogue & Coreference (Anaphora resolution)  
- **Level 4**: Conversational Repair & Ellipsis (Discourse state updates)  
- **Level 5**: Counterfactual Contrastive Drift (Disentangling causal factors)  

| Model Architecture | Level 1 (Direct) | Level 2 (Drift) | Level 3 (Coreference) | Level 4 (Repair/Ellipsis) | Level 5 (Counterfactual) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Inductive Rule Learner | 100.0% | 100.0% | 100.0% | 66.2% | 0.0% |
| Statistical Naive Bayes | 100.0% | 100.0% | 100.0% | 66.2% | 0.0% |
| Majority Baseline | 9.1% | 0.0% | 0.0% | 0.0% | 0.0% |
| Surface Lookup Baseline | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## 4. Adversarial Shortcut Audits & Leakage Resistance

Evaluation of model degradation under adversarial ablations to confirm genuine in-context adaptation:

| Model Architecture | Full Context | Support Ablation (k=0) | Context Ablation | Metadata Only | Permutation Drop | Vulnerability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Majority Baseline | 1.0% | 0.0% | 0.0% | 1.0% | 1.0% | **Robust (Adapting)** |
| Surface Lookup Baseline | 0.0% | 0.0% | 0.0% | 1.0% | 0.0% | **Robust (Adapting)** |
| Statistical Naive Bayes | 75.0% | 18.0% | 2.0% | 1.0% | 57.0% | **Robust (Adapting)** |
| Inductive Rule Learner | 75.0% | 18.0% | 2.0% | 1.0% | 57.0% | **Robust (Adapting)** |

## 5. Model Diagnostics & Error Taxonomy

### Diagnostic Profile: Majority Baseline
- **Total Errors**: 100 / 100 (100.0% error rate)
- **Primary Failure Modes**:
  - `LexicalInductionFailure`: 99 (99.0% of errors)
  - `DiscourseTrackingFailure`: 74 (74.0% of errors)
  - `MorphosyntacticFailure`: 44 (44.0% of errors)
  - `DisentanglementFailure`: 1 (1.0% of errors)
- **Recommendations for Improvement**:
  - High Lexical Induction Failure: Your model is failing to bind novel root morphemes from the support set. Ensure your induction algorithm aligns word order constituents (SOV/SVO/VSO) before extracting mappings.
  - Significant Morphosyntactic Failure: Check your affix segmentation module. Historical time periods introduce phonological lenitions and suffix overrides for tense (-il, -en, -or) and negation prefixes (no-).
  - Discourse Tracking Vulnerability: The model is struggling with coreference and conversational repair. Track speaker discourse state across turns and update salience when a speaker corrects an earlier assertion.
  - Disentanglement Gap: The model recognizes semantic meaning but cannot infer causal origin. Compare the query against the support set baseline: if the lexical mapping differs from the root language, check whether that divergence occurs in other historical periods of that community vs across all communities.

### Diagnostic Profile: Surface Lookup Baseline
- **Total Errors**: 100 / 100 (100.0% error rate)
- **Primary Failure Modes**:
  - `LexicalInductionFailure`: 97 (97.0% of errors)
  - `DiscourseTrackingFailure`: 74 (74.0% of errors)
  - `MorphosyntacticFailure`: 64 (64.0% of errors)
- **Recommendations for Improvement**:
  - High Lexical Induction Failure: Your model is failing to bind novel root morphemes from the support set. Ensure your induction algorithm aligns word order constituents (SOV/SVO/VSO) before extracting mappings.
  - Significant Morphosyntactic Failure: Check your affix segmentation module. Historical time periods introduce phonological lenitions and suffix overrides for tense (-il, -en, -or) and negation prefixes (no-).
  - Discourse Tracking Vulnerability: The model is struggling with coreference and conversational repair. Track speaker discourse state across turns and update salience when a speaker corrects an earlier assertion.

### Diagnostic Profile: Statistical Naive Bayes
- **Total Errors**: 100 / 100 (100.0% error rate)
- **Primary Failure Modes**:
  - `DisentanglementFailure`: 75 (75.0% of errors)
  - `LexicalInductionFailure`: 25 (25.0% of errors)
  - `DiscourseTrackingFailure`: 25 (25.0% of errors)
- **Recommendations for Improvement**:
  - Discourse Tracking Vulnerability: The model is struggling with coreference and conversational repair. Track speaker discourse state across turns and update salience when a speaker corrects an earlier assertion.
  - Disentanglement Gap: The model recognizes semantic meaning but cannot infer causal origin. Compare the query against the support set baseline: if the lexical mapping differs from the root language, check whether that divergence occurs in other historical periods of that community vs across all communities.

### Diagnostic Profile: Inductive Rule Learner
- **Total Errors**: 26 / 100 (26.0% error rate)
- **Primary Failure Modes**:
  - `LexicalInductionFailure`: 25 (96.2% of errors)
  - `DiscourseTrackingFailure`: 25 (96.2% of errors)
  - `DisentanglementFailure`: 1 (3.9% of errors)
- **Recommendations for Improvement**:
  - High Lexical Induction Failure: Your model is failing to bind novel root morphemes from the support set. Ensure your induction algorithm aligns word order constituents (SOV/SVO/VSO) before extracting mappings.
  - Discourse Tracking Vulnerability: The model is struggling with coreference and conversational repair. Track speaker discourse state across turns and update salience when a speaker corrects an earlier assertion.
  - Disentanglement Gap: The model recognizes semantic meaning but cannot infer causal origin. Compare the query against the support set baseline: if the lexical mapping differs from the root language, check whether that divergence occurs in other historical periods of that community vs across all communities.

---  
*Report automatically generated by `shift.evaluation_report.ReportGenerator`.*