# Deliverable 4: Model-Ready IntentAugment AI Pipeline

This directory contains the final model-ready dataset splits, downstream classification benchmark results, and an interactive demonstration for **Deliverable #4 (Week 4 Milestone)** of **IntentAugment AI**.

## Directory Layout

```plaintext
Model-Ready IntentAugment AI Pipeline/
├── data/
│   └── splits/
│       ├── baseline_train.csv          <-- Baseline training set (136 clean samples)
│       ├── augmented_train.csv         <-- Expanded augmented training set (545 samples)
│       ├── validation.csv              <-- Stratified validation set (32 samples)
│       ├── test_clean.csv              <-- Clean evaluation test set (32 samples)
│       └── test_noisy_robustness.csv   <-- Noisy OOD evaluation set (32 samples)
├── benchmarks/
│   └── robustness_benchmark_results.json <-- Benchmark metrics & telemetry
├── demo/
│   └── interactive_demo.py             <-- Real-time CLI interactive comparison demo
├── docs/
│   ├── Model_Ready_IntentAugment_AI_Report.md    <-- Formal technical report
│   └── Model_Ready_IntentAugment_AI_Report.html  <-- Styled reviewable report
└── README.md
```

## Robustness Benchmark Highlights
- **+3.12% Robustness Accuracy Gain** on noisy real-world customer transcripts.
- **+2.65% Macro F1 Improvement** across all 8 business intent categories.
- **Strict Zero Data Leakage** across Train, Validation, and Test partitions.

## Interactive Demonstration
Run the live comparison demo:
```bash
python demo/interactive_demo.py
```
