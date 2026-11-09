# IOAI 2027 Scientific Committee & Jury: Evaluation Package
## Project SHIFT: Semantic Adaptation Across Communities and Historical Time

This package contains the official, deterministic grading infrastructure and secret answer key for evaluating contestant submissions in the IOAI 2027 NLP Practical Round.

---

### Package Contents

1. `hidden_test_labels.json`: The secret ground-truth answer key mapping each private test episode ID (`test_iid_*`, `test_drift_*`, `test_dial_*`, `test_counter_*`) to its canonical `EventFrame` and causal `drift_attribution`.
2. `grade_submission.py`: The official zero-dependency CLI grading script used by the IOAI jury.
3. `README.md`: This reference document.

---

### Scoring Methodology

Submissions are evaluated deterministically using the official **IOAI 2027 Composite Scoring Formula**:

$$\text{Score} = 0.40 \times \text{EM} + 0.25 \times \text{Slot-F1} + 0.15 \times \text{Disentangle} + 0.10 \times \text{RefRes} + 0.10 \times \text{CF-Consist}$$

| Metric Component | Weight | Definition & Purpose |
| :--- | :---: | :--- |
| **Exact Match (EM)** | **40%** | All 5 event frame slots (`agent`, `action`, `object`, `tense`, `polarity`) and `resolved_references` must strictly match ground truth simultaneously. |
| **Macro Slot F1** | **25%** | Partial credit across the 5 event frame slots. Ensures that models correctly predicting agent and action still receive partial marks even if tense fails. |
| **Drift Disentanglement** | **15%** | Measures causal inference accuracy: attributing whether each slot shifted due to `community_shift`, `historical_shift`, `interaction_shift`, or remained `stable`. |
| **Reference Resolution** | **10%** | Accuracy on mapping ambiguous pronouns (`la`, `id`, `te`) and elided antecedents across multi-turn discourse. |
| **Counterfactual Consistency** | **10%** | Penalizes models that output identical memorized answers on paired counterfactual contrastive contexts. |

---

### How to Grade Submissions

From the terminal (using standard Python 3.10+):

```bash
# Grade a contestant submission
python grade_submission.py --submission /path/to/contestant_predictions.json

# Grade and save structured JSON results
python grade_submission.py --submission /path/to/contestant_predictions.json --output results/team_42_score.json
```

### Submission Format Validation
Submissions must be valid JSON matching the official Draft-07 schema:
```json
{
  "test_iid_0001": {
    "agent": "artisan_gamma",
    "action": "inspect",
    "object": "energy_matrix",
    "tense": "future",
    "polarity": "affirmative",
    "resolved_references": {},
    "drift_attribution": {
      "agent": "stable",
      "action": "stable",
      "object": "stable",
      "tense": "stable",
      "polarity": "stable"
    }
  }
}
```
If a contestant submits invalid slot enums or malformed JSON, `grade_submission.py` logs the schema violation and scores that episode as zero.
