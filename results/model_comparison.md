# Project SHIFT: Comparative Benchmark Results

| Model Architecture | Exact Match (%) | Macro Slot F1 (%) | Disentanglement (%) | Reference Res (%) | Counterfactual (%) | Composite Score (/100) | Latency (ms/ep) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 2.0% | 48.6% | 0.0% | 31.0% | 100.0% | **26.1** | 0.015 |
| **Surface Lookup Baseline** | 0.0% | 45.2% | 0.0% | 33.0% | 100.0% | **24.6** | 0.019 |
| **Statistical Naive Bayes** | 86.0% | 97.0% | 0.0% | 90.0% | 100.0% | **77.6** | 0.042 |
| **Inductive Rule Learner** | 86.0% | 97.0% | 97.0% | 90.0% | 100.0% | **92.2** | 0.024 |
