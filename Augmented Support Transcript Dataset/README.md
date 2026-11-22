# Deliverable 3: Augmented Support Transcript Dataset

This directory contains the expanded, validated customer support dataset and quality evaluation reports for **Deliverable #3 (Week 3 Milestone)** of **IntentAugment AI**.

## Directory Structure

```plaintext
Augmented Support Transcript Dataset/
├── data/
│   ├── augmented_support_transcripts.csv   <-- 1,000 augmented instances with full metadata
│   └── full_expanded_training_set.csv      <-- 1,200 total samples (200 raw + 1,000 augmented)
├── reports/
│   ├── Augmented_Dataset_Validation_Report.md    <-- Formal technical validation report
│   ├── Augmented_Dataset_Validation_Report.html  <-- Styled HTML report for review
│   └── dataset_quality_and_validation_report.json <-- Complete numerical telemetry
├── generate_dataset.py                     <-- Reproducible dataset generator script
└── README.md
```

## Key Deliverable Highlights
- **1,000 Augmented Support Transcripts** generated across 5 distinct transformation strategies.
- **1,200 Total Expanded Corpus** (6x dataset expansion over baseline).
- **100% Intent Preservation** verified via dense cosine similarity gating (Mean Similarity: `0.9101`).
- **Perfect Class Balance**: Exactly 150 instances per each of the 8 business intent categories.
- **+50.71% Vocabulary Expansion** introducing real-world colloquialisms, typographical errors, and contextual synonyms.

## Reproduction
To regenerate the dataset and recalculate all quality validation metrics:
```bash
python generate_dataset.py
```
