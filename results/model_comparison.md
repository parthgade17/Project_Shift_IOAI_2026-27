# Project SHIFT: Comparative Benchmark Results

| Model Architecture | Exact Match (%) | Macro Slot F1 (%) | Disentanglement (%) | Reference Res (%) | Counterfactual (%) | Composite Score (/100) | Latency (ms/ep) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 2.0% | 48.4% | 0.0% | 31.0% | 100.0% | **26.0** | 0.017 |
| **Surface Lookup Baseline** | 0.0% | 45.1% | 0.0% | 33.0% | 100.0% | **24.6** | 0.034 |
| **Statistical Naive Bayes** | 86.0% | 97.2% | 0.0% | 90.0% | 100.0% | **77.7** | 0.046 |
| **Inductive Rule Learner** | 86.0% | 97.2% | 97.0% | 90.0% | 100.0% | **92.2** | 0.025 |
