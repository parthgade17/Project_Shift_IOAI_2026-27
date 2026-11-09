# IOAI 2027 Competition Starter Kit: Project SHIFT
## Semantic Adaptation Across Communities and Historical Time

Welcome to the **Project SHIFT** task! In this challenge, your AI model must adapt to unfamiliar linguistic conventions from a small labelled support set, interpret multi-turn conversational context, and predict structured semantic event frames.

---

### Task Summary

Each episode provides:
1. **Community & Period Metadata**: The dialect group (`comm_north`, `comm_coast`, `comm_east`, `comm_citadel`) and historical era (`period_archaic`, `period_classical`, `period_contemporary`).
2. **Word Order**: Canonical constituent ordering (`SOV`, `SVO`, `VSO`).
3. **Support Set**: $k$ labelled pairs of `(utterance, event_frame)` demonstrating the community's vocabulary and grammar.
4. **Dialogue Context**: Prior turns in the conversation.
5. **Target Query Utterance**: An unlabelled utterance you must parse.

Your system must predict the target **EventFrame**:
```json
{
  "agent": "scout_alpha",
  "action": "inspect",
  "object": "crystal_core",
  "tense": "past",
  "polarity": "affirmative",
  "resolved_references": {
    "el": "scout_alpha",
    "id": "crystal_core"
  }
}
```

---

### Starter Kit Contents

- `loader.py`: Stream or load JSONL episode files.
- `submission_template.py`: Template file containing the `SubmissionModel` class with a working starter baseline.
- `local_evaluator.py`: Tool to test and score your model locally against development data.
- `sample_data/sample_dev.jsonl`: 5 sample episodes with ground-truth solutions for quick iteration.

---

### Getting Started

1. **Test the Starter Model**:
   ```bash
   python local_evaluator.py
   ```

2. **Evaluate on Public Dev Set**:
   ```bash
   python local_evaluator.py --data ../datasets/public_dev.jsonl
   ```

3. **Improve Your Model**:
   Edit `submission_template.py`:
   - Enhance the morphological aligner.
   - Refine discourse antecedent tracking for pronouns and ellipsis.
   - Implement statistical induction or neural prompt learning.

---

### Evaluation Criteria

- **Primary Metric**: Exact Event-Frame Match Accuracy (all slots + resolved references).
- **Secondary Metrics**: Slot-level accuracies (`agent`, `action`, `object`, `tense`, `polarity`) and latency per query.
- **Constraints**: Pure inference time must not exceed 2.0 seconds per episode. No external internet access during hidden test evaluation.
