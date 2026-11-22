# Deliverable 1: Support Transcript Dataset & Augmentation Specification

This directory contains all required artifacts for **Deliverable #1 (Week 1 Milestone)** of the **IntentAugment AI** project.

## Directory Contents

- `raw_customer_support_transcripts.csv`: Curated baseline customer support dataset featuring 200 verified utterances across 8 balanced business intents (`refund_request`, `cancel_order`, `track_shipping`, `payment_issue`, `account_recovery`, `technical_support`, `subscription_management`, `product_inquiry`).
- `dataset_analysis.py`: Standalone Python script that computes exploratory data analysis (EDA), lexical diversity (TTR), token lengths, and intent class distributions.
- `dataset_eda_summary.json`: Exported structured metrics generated from running `dataset_analysis.py`.
- `Support_Transcript_Dataset_and_Augmentation_Specification.md`: The formal technical specification document covering:
  - Executive summary and problem formulation
  - Baseline dataset characterization
  - Customer phrasing and noise taxonomy (QWERTY typos, acoustic slips, syntactic compression)
  - Augmentation requirements, noise budgets, and semantic quality thresholds (cosine similarity >= 0.80)
  - NLP environment setup (NLPAug, spaCy, Hugging Face Transformers)
  - Modular pipeline architecture and downstream evaluation plan.

## Running the EDA Script
To re-run the exploratory analysis locally:
```bash
python dataset_analysis.py
```
