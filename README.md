# IntentAugment AI: Support Transcript Augmenter

**IntentAugment AI** is an advanced NLP pipeline engineered to bolster the robustness and generalization of intent-classification models. By generating controlled paraphrases and realistic customer noise variations (keyboard typos, acoustic slips, grammatical compression), IntentAugment AI prevents model degradation on real-world customer support transcripts while strictly preserving intent labels.

---

## Project Structure & Deliverables

```plaintext
intentaugment-ai/
├── Support Transcript Dataset & Augmentation Specification/  <-- Deliverable #1 (Week 1)
│   ├── raw_customer_support_transcripts.csv                  <-- Curated 200 baseline transcripts
│   ├── dataset_analysis.py                                   <-- Exploratory Data Analysis script
│   ├── dataset_eda_summary.json                              <-- Computed lexical & structural metrics
│   ├── Support_Transcript_Dataset_and_Augmentation_Specification.md <-- Formal Specification
│   └── README.md
├── Transcript Paraphrasing & Noise Augmentation Pipeline/     <-- Deliverable #2 (Week 2)
│   └── info.txt
├── Augmented Support Transcript Dataset/                     <-- Deliverable #3 (Week 3)
│   └── info.txt
├── Model-Ready IntentAugment AI Pipeline/                    <-- Deliverable #4 (Week 4)
│   └── info.txt
├── requirements.txt
└── README.md
```

---

## Core Capabilities
1. **Transformer Paraphrasing:** Contextual paraphrasing using Hugging Face sequence-to-sequence models (T5/Pegasus).
2. **Realistic Noise Injection:** QWERTY keyboard distance typos and OCR perturbation using `NLPAug`.
3. **Linguistic Compression:** Grammatical and syntactic variation using `spaCy`.
4. **Intent Preservation Gate:** Dense semantic similarity verification (>= 0.80 cosine similarity) via `sentence-transformers`.
5. **Downstream Benchmarking:** Robustness testing on noisy customer support evaluations.

---

## Quickstart

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Deliverable #1 Dataset Analysis
```bash
cd "Support Transcript Dataset & Augmentation Specification"
python dataset_analysis.py
```
