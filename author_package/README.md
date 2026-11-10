# Project SHIFT: Author & Scientific Committee Release Package
**International Olympiad in Artificial Intelligence (IOAI 2027)**  
*Task Category: Natural Language Processing — Few-Shot In-Context Semantic Adaptation & Drift Disentanglement*

---

## 1. Overview for Authors & Jury
This package is strictly reserved for the **IOAI 2027 Scientific Committee & Official Jury**. It contains the authoritative benchmark generators, ground-truth answer keys, adversarial shortcut audit suites, and official automated scoring infrastructure.

> **CRITICAL SECURITY PROTOCOL**:  
> Files in `private_data/` (`hidden_test_labels.json`, `hidden_test_full_reference.jsonl`) must **NEVER** be shared with contestants or published prior to the conclusion of the Olympiad.

---

## 2. Directory Structure
```
author_package/
├── README.md                   # This document
├── generate_benchmarks.py      # Master seed-controlled dataset generator
├── grade_submission.py         # Official jury grading tool
├── shortcut_auditor.py         # Adversarial shortcut verification suite
└── private_data/               # Secret ground-truth keys
    ├── hidden_test_labels.json
    └── hidden_test_full_reference.jsonl
```

---

## 3. Official Jury Grading Pipeline
To evaluate a contestant's submission file (`predictions.json` or `predictions.jsonl`):

```bash
python author_package/grade_submission.py --submission path/to/contestant_submission.json
```

### Official Scoring Metric Breakdown:
- **Exact Match (EM)** (40% weight): All 5 semantic slots (`agent`, `action`, `object`, `tense`, `polarity`) must exactly match ground truth.
- **Slot-Level Macro F1** (25% weight): Partial credit across semantic constituents.
- **Drift Disentanglement** (15% weight): Correct attribution of community vs historical shift.
- **Reference Resolution** (10% weight): Accurate resolution of pronouns, ellipsis, and dialogue corrections.
- **Counterfactual Consistency** (10% weight): Sensitivity to context perturbations in paired counterfactual queries.

---

## 4. Benchmark Quality & Verification Gates
Before deploying a new version of the benchmark, run:
```bash
python author_package/shortcut_auditor.py
```
This certifies:
1. **Zero Answer Leakage**: Dialogue context contains zero gold event frames.
2. **100% Learnability**: All 6 formal checks passed + contestant blind solver certified.
3. **Shortcut Resistance**: Metadata-only attack accuracy $\le 15\%$, support ablation $k=0$ drop confirmed.
