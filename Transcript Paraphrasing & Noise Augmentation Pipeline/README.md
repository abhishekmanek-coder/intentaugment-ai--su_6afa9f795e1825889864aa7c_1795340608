# Deliverable 2: Transcript Paraphrasing & Noise Augmentation Pipeline

This directory contains all source code for **Deliverable #2 (Week 2 Milestone)** of the **IntentAugment AI** project.

## Overview

A working NLP augmentation pipeline that implements controlled paraphrasing and noise injection for customer support transcripts using NLPAug, spaCy, and Hugging Face Transformers — while preserving the original customer intent.

## Directory Structure

```plaintext
Transcript Paraphrasing & Noise Augmentation Pipeline/
├── src/
│   ├── __init__.py              # Package initializer
│   ├── dataset_loader.py        # Data ingestion and validation (CSV/JSON/JSONL)
│   ├── paraphraser.py           # Transformer & rule-based paraphrase generation
│   ├── noise_generator.py       # NLPAug keyboard/OCR/char noise + spaCy linguistic noise
│   ├── validator.py             # Sentence-Transformers semantic similarity validation
│   └── pipeline.py              # Unified orchestrator with CLI interface
├── configs/
│   └── augmentation_config.yaml # Full pipeline configuration (noise rates, thresholds, models)
├── tests/
│   └── test_pipeline.py         # Unit and integration tests for all modules
└── README.md
```

## Module Descriptions

### `src/dataset_loader.py` — Data Ingestion
- Loads raw customer support transcripts from CSV, JSON, or JSONL files.
- Validates required fields (text, intent label) and filters empty/invalid records.
- Computes label distribution for class-balance verification.

### `src/paraphraser.py` — Contextual Paraphrasing
- **TranscriptParaphraser**: Full Hugging Face T5-based paraphraser with configurable beam search, temperature, top-p/top-k sampling, and repetition penalty.
- **RuleBasedParaphraser**: Lightweight fallback using domain-specific synonym dictionaries, contraction toggling, and politeness modulation — requires no model downloads.
- Both produce multiple diverse variants per input while deduplicating outputs.

### `src/noise_generator.py` — Noise & Typo Injection
Implements four noise strategies:
1. **Keyboard Noise** (`nlpaug.augmenter.char.KeyboardAug`): QWERTY-adjacent key substitutions simulating fat-finger typos.
2. **OCR Noise** (`nlpaug.augmenter.char.OcrAug`): Character confusions from optical recognition (l↔1, O↔0, S↔5).
3. **Character Noise** (`nlpaug.augmenter.char.RandomCharAug`): Random insertions, deletions, swaps, and substitutions.
4. **Linguistic Noise**: Domain-aware synonym replacement, contraction expansion/contraction, and case variation.
5. **Combined Noise**: Stacks multiple noise types for maximum diversity.

All noise is budget-constrained: max 12% character perturbation and 15% word perturbation rate.
Built-in fallback implementations work without NLPAug/spaCy installed.

### `src/validator.py` — Semantic Intent Preservation
- Uses `sentence-transformers` (`all-MiniLM-L6-v2`) to compute dense cosine similarity.
- Acceptance window: similarity ∈ [0.80, 0.98].
- Below 0.80 → rejected as semantic drift (meaning changed too much).
- Above 0.98 → rejected as trivial copy (insufficient variation).
- Includes fallback word-overlap + character n-gram similarity when SBERT is unavailable.

### `src/pipeline.py` — Unified Pipeline Orchestrator
- Integrates all components into a single `AugmentationPipeline` class.
- Configurable strategies, variant counts, and validation thresholds.
- Full CLI interface with argument parsing.
- Exports augmented dataset to CSV with metadata tracking (strategy, similarity score).
- Exports execution statistics to JSON.

## Quick Start

### Option A: Run with built-in fallbacks (no downloads needed)
```bash
cd src
python pipeline.py --input "../../Support Transcript Dataset & Augmentation Specification/raw_customer_support_transcripts.csv" --output augmented_output.csv --no-validation
```

### Option B: Run with full NLP stack
```bash
# Install dependencies
pip install transformers torch nlpaug spacy sentence-transformers
python -m spacy download en_core_web_sm

# Run with transformer paraphraser and semantic validation
cd src
python pipeline.py --input "../../Support Transcript Dataset & Augmentation Specification/raw_customer_support_transcripts.csv" --output augmented_output.csv --transformer
```

### CLI Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--input`, `-i` | Auto-detect | Path to input transcript CSV |
| `--output`, `-o` | `augmented_transcripts.csv` | Output augmented dataset path |
| `--stats`, `-s` | `pipeline_stats.json` | Output statistics JSON path |
| `--variants`, `-n` | `4` | Number of variants per transcript |
| `--transformer` | `False` | Use HF transformer paraphraser |
| `--no-validation` | `False` | Disable semantic similarity validation |
| `--seed` | `42` | Random seed for reproducibility |
| `--sim-floor` | `0.80` | Minimum cosine similarity threshold |
| `--sim-ceiling` | `0.98` | Maximum cosine similarity threshold |

## Running Tests
```bash
cd tests
python test_pipeline.py
```

Expected output: All tests should pass, covering DatasetLoader, Paraphraser, NoiseGenerator, Validator, and end-to-end pipeline integration.
