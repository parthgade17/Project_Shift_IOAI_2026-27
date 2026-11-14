# Project SHIFT: Official Benchmark Results (IOAI 2027)

**Benchmark Version**: Project SHIFT v0.3 (IOAI 2027 Scientific Benchmark Infrastructure)  
**Target Domain**: In-Context Semantic Adaptation & Drift Disentanglement  

---

## 1. Overall Model Performance Comparison

The table below compares baseline models on exact event-frame accuracy, the official IOAI composite score (partial credit), semantic drift disentanglement, reference resolution, and counterfactual consistency.

| Model Architecture | Exact Match (%) | IOAI Composite Score | Disentanglement (%) | Reference Res. (%) | Counterfactual Consist. (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Inductive Rule Learner** | 86.00% | **92.19** | 97.00% | 90.00% | 100.00% |
| **Statistical Naive Bayes** | 86.00% | **77.64** | 0.00% | 90.00% | 100.00% |
| **Majority Baseline** | 2.00% | **26.05** | 0.00% | 31.00% | 100.00% |
| **Surface Lookup Baseline** | 0.00% | **24.60** | 0.00% | 33.00% | 100.00% |

## 2. Slot-Level Macro F1 Breakdown

Macro F1 scores across individual semantic frame slots:

| Model Architecture | Agent F1 | Action F1 | Object F1 | Tense F1 | Polarity F1 | Mean Slot F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Inductive Rule Learner | 95.93 | 100.00 | 88.90 | 100.00 | 100.00 | **96.97** |
| Statistical Naive Bayes | 95.93 | 100.00 | 88.90 | 100.00 | 100.00 | **96.97** |
| Majority Baseline | 52.48 | 36.46 | 25.71 | 76.62 | 51.78 | **48.61** |
| Surface Lookup Baseline | 58.17 | 22.30 | 23.31 | 65.90 | 56.38 | **45.21** |

## 3. Progressive Difficulty Levels (Curriculum Evaluation)

- **Level 1**: Direct Canonical Observation (Isolated context, fixed syntax)  
- **Level 2**: Unseen Semantic Drift (Factorial community/historical shift)  
- **Level 3**: Multi-Turn Dialogue & Coreference (Anaphora resolution)  
- **Level 4**: Conversational Repair & Ellipsis (Discourse state updates)  
- **Level 5**: Counterfactual Contrastive Drift (Disentangling causal factors)  

| Model Architecture | Level 1 (Direct) | Level 2 (Drift) | Level 3 (Coreference) | Level 4 (Repair/Ellipsis) | Level 5 (Counterfactual) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Inductive Rule Learner | 100.0% | 91.7% | 100.0% | 70.5% | 100.0% |
| Statistical Naive Bayes | 100.0% | 91.7% | 100.0% | 70.5% | 100.0% |
| Majority Baseline | 14.3% | 8.3% | 0.0% | 0.0% | 0.0% |
| Surface Lookup Baseline | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## 4. Adversarial Shortcut Audits & Leakage Resistance

Evaluation of model degradation under adversarial ablations to confirm genuine in-context adaptation:

| Model Architecture | Full Context | Support Ablation (k=0) | Context Ablation | Metadata Only | Permutation Drop | Vulnerability |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Majority Baseline | 2.0% | 0.0% | 0.0% | 10.0% | 2.0% | **Robust (Adapting)** |
| Surface Lookup Baseline | 0.0% | 0.0% | 0.0% | 10.0% | 0.0% | **Robust (Adapting)** |
| Statistical Naive Bayes | 86.0% | 15.0% | 0.0% | 10.0% | 71.0% | **Robust (Adapting)** |
| Inductive Rule Learner | 86.0% | 15.0% | 2.2% | 10.0% | 71.0% | **Robust (Adapting)** |

## 5. Model Diagnostics & Error Taxonomy

### Diagnostic Profile: Majority Baseline
- **Total Errors**: 100 / 100 (100.0% error rate)
- **Primary Failure Modes**:
  - `LexicalInductionFailure`: 95 (95.0% of errors)
  - `DiscourseTrackingFailure`: 69 (69.0% of errors)
  - `MorphosyntacticFailure`: 32 (32.0% of errors)
  - `DisentanglementFailure`: 2 (2.0% of errors)
- **Recommendations for Improvement**:
  - High Lexical Induction Failure: Your model is failing to bind novel root morphemes from the support set. Ensure your induction algorithm aligns word order constituents (SOV/SVO/VSO) before extracting mappings.
  - Significant Morphosyntactic Failure: Check your affix segmentation module. Historical time periods introduce phonological lenitions and suffix overrides for tense (-il, -en, -or) and negation prefixes (no-).
  - Discourse Tracking Vulnerability: The model is struggling with coreference and conversational repair. Track speaker discourse state across turns and update salience when a speaker corrects an earlier assertion.
  - Disentanglement Gap: The model recognizes semantic meaning but cannot infer causal origin. Compare the query against the support set baseline: if the lexical mapping differs from the root language, check whether that divergence occurs in other historical periods of that community vs across all communities.

### Diagnostic Profile: Surface Lookup Baseline
- **Total Errors**: 100 / 100 (100.0% error rate)
- **Primary Failure Modes**:
  - `LexicalInductionFailure`: 99 (99.0% of errors)
  - `DiscourseTrackingFailure`: 69 (69.0% of errors)
  - `MorphosyntacticFailure`: 47 (47.0% of errors)
- **Recommendations for Improvement**:
  - High Lexical Induction Failure: Your model is failing to bind novel root morphemes from the support set. Ensure your induction algorithm aligns word order constituents (SOV/SVO/VSO) before extracting mappings.
  - Significant Morphosyntactic Failure: Check your affix segmentation module. Historical time periods introduce phonological lenitions and suffix overrides for tense (-il, -en, -or) and negation prefixes (no-).
  - Discourse Tracking Vulnerability: The model is struggling with coreference and conversational repair. Track speaker discourse state across turns and update salience when a speaker corrects an earlier assertion.

### Diagnostic Profile: Statistical Naive Bayes
- **Total Errors**: 100 / 100 (100.0% error rate)
- **Primary Failure Modes**:
  - `DisentanglementFailure`: 86 (86.0% of errors)
  - `DiscourseTrackingFailure`: 14 (14.0% of errors)
  - `LexicalInductionFailure`: 10 (10.0% of errors)
- **Recommendations for Improvement**:
  - Disentanglement Gap: The model recognizes semantic meaning but cannot infer causal origin. Compare the query against the support set baseline: if the lexical mapping differs from the root language, check whether that divergence occurs in other historical periods of that community vs across all communities.

### Diagnostic Profile: Inductive Rule Learner
- **Total Errors**: 15 / 100 (15.0% error rate)
- **Primary Failure Modes**:
  - `DiscourseTrackingFailure`: 14 (93.3% of errors)
  - `LexicalInductionFailure`: 10 (66.7% of errors)
  - `DisentanglementFailure`: 1 (6.7% of errors)
- **Recommendations for Improvement**:
  - High Lexical Induction Failure: Your model is failing to bind novel root morphemes from the support set. Ensure your induction algorithm aligns word order constituents (SOV/SVO/VSO) before extracting mappings.
  - Discourse Tracking Vulnerability: The model is struggling with coreference and conversational repair. Track speaker discourse state across turns and update salience when a speaker corrects an earlier assertion.
  - Disentanglement Gap: The model recognizes semantic meaning but cannot infer causal origin. Compare the query against the support set baseline: if the lexical mapping differs from the root language, check whether that divergence occurs in other historical periods of that community vs across all communities.

---  
*Report automatically generated by `shift.evaluation_report.ReportGenerator`.*