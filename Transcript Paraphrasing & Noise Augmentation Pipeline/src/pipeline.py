"""
IntentAugment AI - Unified Augmentation Pipeline
==================================================
Orchestrates the complete transcript augmentation workflow by combining
paraphrasing (Hugging Face Transformers), noise injection (NLPAug),
linguistic perturbation (spaCy), and semantic validation
(Sentence-Transformers) into a single reusable pipeline.

Author: Abhishek Manek
Deliverable: #2 (Week 2) - Transcript Paraphrasing & Noise Augmentation Pipeline
"""

import os
import csv
import json
import time
import random
from typing import List, Dict, Any, Optional
from datetime import datetime

# Local module imports
from dataset_loader import DatasetLoader
from paraphraser import TranscriptParaphraser, RuleBasedParaphraser
from noise_generator import NoiseGenerator
from validator import SemanticValidator


class AugmentationPipeline:
    """
    Unified augmentation pipeline that orchestrates all IntentAugment AI
    components to generate diverse, intent-preserving customer support
    transcript variations.
    
    Pipeline Flow:
        1. Load & validate raw support transcripts (DatasetLoader)
        2. Apply paraphrasing strategies (TranscriptParaphraser / RuleBasedParaphraser)
        3. Apply noise injection strategies (NoiseGenerator)
        4. Validate semantic integrity (SemanticValidator)
        5. Export augmented dataset with metadata tracking
    
    Example usage:
        >>> pipeline = AugmentationPipeline(config_path="configs/augmentation_config.yaml")
        >>> results = pipeline.run(input_path="data/raw/transcripts.csv")
        >>> pipeline.export(results, output_path="data/augmented/augmented_transcripts.csv")
    """
    
    # Available augmentation strategy names
    STRATEGY_PARAPHRASE = "paraphrase"
    STRATEGY_KEYBOARD = "keyboard_noise"
    STRATEGY_OCR = "ocr_noise"
    STRATEGY_CHAR = "char_noise"
    STRATEGY_LINGUISTIC = "linguistic_noise"
    STRATEGY_COMBINED = "combined"
    
    ALL_STRATEGIES = [
        STRATEGY_PARAPHRASE, STRATEGY_KEYBOARD, STRATEGY_OCR,
        STRATEGY_CHAR, STRATEGY_LINGUISTIC, STRATEGY_COMBINED
    ]
    
    def __init__(
        self,
        num_variants: int = 4,
        strategies: Optional[List[str]] = None,
        use_transformer_paraphraser: bool = False,
        transformer_model: str = "Vamsi/T5_Paraphrase_Paws",
        seed: int = 42,
        similarity_floor: float = 0.80,
        similarity_ceiling: float = 0.98,
        enable_validation: bool = True,
        config_path: Optional[str] = None
    ):
        """
        Initialize the AugmentationPipeline.
        
        Args:
            num_variants: Number of augmented variants to generate per transcript.
            strategies: List of augmentation strategies to apply.
            use_transformer_paraphraser: Whether to use HF transformer (requires download).
            transformer_model: Hugging Face model name for paraphrasing.
            seed: Global random seed for reproducibility.
            similarity_floor: Minimum cosine similarity for validation.
            similarity_ceiling: Maximum cosine similarity for validation.
            enable_validation: Whether to run semantic validation filtering.
            config_path: Optional path to YAML configuration file.
        """
        self.num_variants = num_variants
        self.strategies = strategies or self.ALL_STRATEGIES
        self.use_transformer = use_transformer_paraphraser
        self.seed = seed
        self.enable_validation = enable_validation
        
        random.seed(seed)
        
        # Initialize components
        print("[Pipeline] Initializing IntentAugment AI Pipeline...")
        print(f"[Pipeline] Seed: {seed}")
        print(f"[Pipeline] Strategies: {self.strategies}")
        print(f"[Pipeline] Variants per sample: {num_variants}")
        
        # 1. Dataset Loader
        self.loader = DatasetLoader()
        
        # 2. Paraphraser
        if use_transformer_paraphraser:
            self.paraphraser = TranscriptParaphraser(
                model_name=transformer_model,
                seed=seed
            )
            print("[Pipeline] Using Transformer paraphraser (model download required)")
        else:
            self.paraphraser = RuleBasedParaphraser(seed=seed)
            print("[Pipeline] Using Rule-Based paraphraser (no download needed)")
        
        # 3. Noise Generator
        self.noise_gen = NoiseGenerator(seed=seed)
        
        # 4. Semantic Validator
        self.validator = SemanticValidator(
            similarity_floor=similarity_floor,
            similarity_ceiling=similarity_ceiling
        )
        
        # Statistics tracking
        self.stats = {
            "total_originals": 0,
            "total_generated": 0,
            "total_accepted": 0,
            "total_rejected": 0,
            "strategy_counts": {},
            "rejection_reasons": {"semantic_drift": 0, "trivial_copy": 0},
            "processing_time_seconds": 0
        }
        
        print("[Pipeline] Initialization complete.\n")
    
    def run(
        self,
        input_path: Optional[str] = None,
        records: Optional[List[Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute the full augmentation pipeline.
        
        Args:
            input_path: Path to raw transcript CSV/JSON file.
            records: Pre-loaded list of transcript records (alternative to input_path).
            
        Returns:
            List of augmented record dictionaries with metadata.
        """
        start_time = time.time()
        
        # Step 1: Load data
        if records is None:
            if input_path is None:
                raise ValueError("Either input_path or records must be provided.")
            records = self.loader.load(input_path)
        
        self.stats["total_originals"] = len(records)
        print(f"[Pipeline] Processing {len(records)} transcripts...")
        
        # Step 2: Try loading validation model
        if self.enable_validation:
            try:
                self.validator.load_model()
            except Exception as e:
                print(f"[Pipeline] Validation model not available: {e}")
                print("[Pipeline] Using fallback word-overlap similarity.")
        
        # Step 3: Generate augmented variants
        all_augmented = []
        
        for idx, record in enumerate(records):
            original_text = record["text"]
            intent = record["intent"]
            transcript_id = record["transcript_id"]
            
            # Generate variants using different strategies
            variants = self._generate_variants(original_text)
            
            # Step 4: Validate each variant
            for variant_info in variants:
                augmented_text = variant_info["text"]
                strategy = variant_info["strategy"]
                
                # Semantic validation
                if self.enable_validation:
                    validation = self.validator.validate_variant(original_text, augmented_text)
                    similarity = validation["similarity"]
                    is_valid = validation["is_valid"]
                    rejection_reason = validation["rejection_reason"]
                else:
                    similarity = -1.0
                    is_valid = True
                    rejection_reason = None
                
                self.stats["total_generated"] += 1
                
                if is_valid:
                    self.stats["total_accepted"] += 1
                    augmented_record = {
                        "original_id": transcript_id,
                        "original_text": original_text,
                        "augmented_text": augmented_text,
                        "intent": intent,
                        "augmentation_strategy": strategy,
                        "similarity_score": similarity,
                        "is_valid": True,
                        "channel": record.get("channel", ""),
                        "sentiment": record.get("sentiment", ""),
                    }
                    all_augmented.append(augmented_record)
                else:
                    self.stats["total_rejected"] += 1
                    if rejection_reason and "drift" in rejection_reason:
                        self.stats["rejection_reasons"]["semantic_drift"] += 1
                    elif rejection_reason and "trivial" in rejection_reason:
                        self.stats["rejection_reasons"]["trivial_copy"] += 1
                
                # Track strategy counts
                self.stats["strategy_counts"][strategy] = \
                    self.stats["strategy_counts"].get(strategy, 0) + 1
            
            # Progress reporting
            if (idx + 1) % 50 == 0 or idx == len(records) - 1:
                print(f"[Pipeline] Processed {idx + 1}/{len(records)} transcripts "
                      f"({len(all_augmented)} accepted variants so far)")
        
        elapsed = round(time.time() - start_time, 2)
        self.stats["processing_time_seconds"] = elapsed
        
        self._print_summary()
        
        return all_augmented
    
    def _generate_variants(self, text: str) -> List[Dict[str, str]]:
        """
        Generate augmented variants using configured strategies.
        
        Distributes the requested number of variants across the
        enabled strategies, applying each one to generate diverse outputs.
        
        Args:
            text: Original customer utterance.
            
        Returns:
            List of dicts with 'text' and 'strategy' keys.
        """
        variants = []
        variants_per_strategy = max(1, self.num_variants // len(self.strategies))
        
        for strategy in self.strategies:
            for _ in range(variants_per_strategy):
                try:
                    augmented = self._apply_strategy(strategy, text)
                    if augmented and augmented.strip() != text.strip():
                        variants.append({
                            "text": augmented,
                            "strategy": strategy
                        })
                except Exception as e:
                    # Log but don't crash on individual failures
                    pass
            
            if len(variants) >= self.num_variants:
                break
        
        return variants[:self.num_variants]
    
    def _apply_strategy(self, strategy: str, text: str) -> str:
        """
        Apply a specific augmentation strategy to text.
        
        Args:
            strategy: Name of the strategy to apply.
            text: Original customer utterance.
            
        Returns:
            Augmented text string.
        """
        if strategy == self.STRATEGY_PARAPHRASE:
            results = self.paraphraser.paraphrase(text, num_variants=1)
            return results[0] if results else text
        
        elif strategy == self.STRATEGY_KEYBOARD:
            return self.noise_gen.apply_keyboard_noise(text)
        
        elif strategy == self.STRATEGY_OCR:
            return self.noise_gen.apply_ocr_noise(text)
        
        elif strategy == self.STRATEGY_CHAR:
            return self.noise_gen.apply_char_noise(text)
        
        elif strategy == self.STRATEGY_LINGUISTIC:
            return self.noise_gen.apply_linguistic_noise(text)
        
        elif strategy == self.STRATEGY_COMBINED:
            return self.noise_gen.apply_combined_noise(text)
        
        else:
            raise ValueError(f"Unknown augmentation strategy: {strategy}")
    
    def export(
        self,
        augmented_records: List[Dict],
        output_path: str,
        include_originals: bool = True
    ):
        """
        Export augmented dataset to CSV file.
        
        Args:
            augmented_records: List of augmented record dictionaries.
            output_path: Output file path (.csv).
            include_originals: Whether to include original transcripts in output.
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        fieldnames = [
            "original_id", "original_text", "augmented_text", "intent",
            "augmentation_strategy", "similarity_score", "is_valid",
            "channel", "sentiment"
        ]
        
        with open(output_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in augmented_records:
                row = {k: record.get(k, "") for k in fieldnames}
                writer.writerow(row)
        
        print(f"[Pipeline] Exported {len(augmented_records)} augmented transcripts to {output_path}")
    
    def export_stats(self, output_path: str):
        """
        Export pipeline execution statistics to JSON.
        
        Args:
            output_path: Output JSON file path.
        """
        stats_with_meta = {
            "pipeline_name": "IntentAugment AI",
            "execution_timestamp": datetime.now().isoformat(),
            **self.stats,
            "acceptance_rate": round(
                self.stats["total_accepted"] / max(1, self.stats["total_generated"]) * 100, 2
            ),
            "expansion_factor": round(
                self.stats["total_accepted"] / max(1, self.stats["total_originals"]), 2
            )
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats_with_meta, f, indent=2)
        
        print(f"[Pipeline] Statistics exported to {output_path}")
    
    def _print_summary(self):
        """Print a formatted summary of the pipeline execution."""
        s = self.stats
        total_gen = max(1, s["total_generated"])
        
        print("\n" + "=" * 60)
        print("  IntentAugment AI - Pipeline Execution Summary")
        print("=" * 60)
        print(f"  Original Transcripts     : {s['total_originals']}")
        print(f"  Total Variants Generated : {s['total_generated']}")
        print(f"  Accepted (Valid)         : {s['total_accepted']} "
              f"({s['total_accepted'] / total_gen * 100:.1f}%)")
        print(f"  Rejected                 : {s['total_rejected']} "
              f"({s['total_rejected'] / total_gen * 100:.1f}%)")
        print(f"    - Semantic Drift       : {s['rejection_reasons']['semantic_drift']}")
        print(f"    - Trivial Copy         : {s['rejection_reasons']['trivial_copy']}")
        print(f"  Expansion Factor         : {s['total_accepted'] / max(1, s['total_originals']):.2f}x")
        print(f"  Processing Time          : {s['processing_time_seconds']:.2f} seconds")
        print()
        print("  Strategy Breakdown:")
        for strategy, count in sorted(s["strategy_counts"].items()):
            print(f"    {strategy:<20}: {count} variants")
        print("=" * 60)
    
    def get_config(self) -> Dict[str, Any]:
        """Return complete pipeline configuration."""
        return {
            "num_variants": self.num_variants,
            "strategies": self.strategies,
            "use_transformer": self.use_transformer,
            "seed": self.seed,
            "enable_validation": self.enable_validation,
            "paraphraser": self.paraphraser.get_config(),
            "noise_generator": self.noise_gen.get_config(),
            "validator": self.validator.get_config()
        }


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """
    Command-line interface for the IntentAugment AI pipeline.
    
    Usage:
        python pipeline.py [--input PATH] [--output PATH] [--variants N] [--transformer]
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="IntentAugment AI: Support Transcript Augmentation Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Path to input transcript CSV file."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="augmented_transcripts.csv",
        help="Path for output augmented dataset CSV."
    )
    parser.add_argument(
        "--stats", "-s",
        type=str,
        default="pipeline_stats.json",
        help="Path for output pipeline statistics JSON."
    )
    parser.add_argument(
        "--variants", "-n",
        type=int,
        default=4,
        help="Number of augmented variants per transcript (default: 4)."
    )
    parser.add_argument(
        "--transformer",
        action="store_true",
        help="Use Hugging Face transformer paraphraser (requires model download)."
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Disable semantic similarity validation."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    parser.add_argument(
        "--sim-floor",
        type=float,
        default=0.80,
        help="Minimum cosine similarity threshold (default: 0.80)."
    )
    parser.add_argument(
        "--sim-ceiling",
        type=float,
        default=0.98,
        help="Maximum cosine similarity threshold (default: 0.98)."
    )
    
    args = parser.parse_args()
    
    # Resolve default input path
    if args.input is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.input = os.path.join(
            script_dir, "..", "..",
            "Support Transcript Dataset & Augmentation Specification",
            "raw_customer_support_transcripts.csv"
        )
    
    # Initialize and run pipeline
    pipeline = AugmentationPipeline(
        num_variants=args.variants,
        use_transformer_paraphraser=args.transformer,
        seed=args.seed,
        similarity_floor=args.sim_floor,
        similarity_ceiling=args.sim_ceiling,
        enable_validation=not args.no_validation
    )
    
    results = pipeline.run(input_path=args.input)
    
    # Export results
    pipeline.export(results, output_path=args.output)
    pipeline.export_stats(output_path=args.stats)
    
    print(f"\n[Pipeline] Done! Augmented dataset: {args.output}")
    print(f"[Pipeline] Statistics: {args.stats}")


if __name__ == "__main__":
    main()
