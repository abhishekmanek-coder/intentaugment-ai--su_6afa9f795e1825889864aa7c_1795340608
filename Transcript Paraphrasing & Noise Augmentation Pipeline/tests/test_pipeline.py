"""
IntentAugment AI - Pipeline Unit Tests
========================================
Tests for the core augmentation pipeline components:
    - DatasetLoader validation
    - RuleBasedParaphraser variant generation
    - NoiseGenerator noise injection
    - SemanticValidator similarity computation
    - AugmentationPipeline end-to-end execution

Author: Abhishek Manek
Run with: python -m pytest tests/test_pipeline.py -v
Or:       python tests/test_pipeline.py
"""

import os
import sys
import csv
import tempfile

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dataset_loader import DatasetLoader
from paraphraser import RuleBasedParaphraser
from noise_generator import NoiseGenerator
from validator import SemanticValidator


# =============================================================================
# Test Data
# =============================================================================

SAMPLE_TRANSCRIPTS = [
    {"transcript_id": "T001", "customer_utterance": "I need a refund for my damaged order", "intent": "refund_request", "channel": "live_chat", "sentiment": "frustrated", "complexity": "simple"},
    {"transcript_id": "T002", "customer_utterance": "Can you cancel my order immediately?", "intent": "cancel_order", "channel": "live_chat", "sentiment": "urgent", "complexity": "simple"},
    {"transcript_id": "T003", "customer_utterance": "Where is my package? It hasn't arrived yet.", "intent": "track_shipping", "channel": "live_chat", "sentiment": "frustrated", "complexity": "compound"},
    {"transcript_id": "T004", "customer_utterance": "My payment was declined at checkout", "intent": "payment_issue", "channel": "live_chat", "sentiment": "frustrated", "complexity": "simple"},
    {"transcript_id": "T005", "customer_utterance": "I can't login to my account, password reset isn't working", "intent": "account_recovery", "channel": "live_chat", "sentiment": "frustrated", "complexity": "compound"},
]


def _create_temp_csv(records):
    """Create a temporary CSV file from records."""
    fd, path = tempfile.mkstemp(suffix=".csv")
    with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    return path


# =============================================================================
# DatasetLoader Tests
# =============================================================================

class TestDatasetLoader:
    
    def test_load_csv(self):
        """Test loading a valid CSV dataset."""
        path = _create_temp_csv(SAMPLE_TRANSCRIPTS)
        try:
            loader = DatasetLoader()
            records = loader.load(path)
            assert len(records) == 5, f"Expected 5 records, got {len(records)}"
            assert records[0]["text"] == "I need a refund for my damaged order"
            assert records[0]["intent"] == "refund_request"
            print("  PASS: test_load_csv")
        finally:
            os.unlink(path)
    
    def test_label_distribution(self):
        """Test label distribution computation."""
        path = _create_temp_csv(SAMPLE_TRANSCRIPTS)
        try:
            loader = DatasetLoader()
            records = loader.load(path)
            dist = loader.get_label_distribution(records)
            assert len(dist) == 5, f"Expected 5 intents, got {len(dist)}"
            for intent, count in dist.items():
                assert count == 1, f"Expected 1 per intent, got {count} for {intent}"
            print("  PASS: test_label_distribution")
        finally:
            os.unlink(path)
    
    def test_empty_text_filtered(self):
        """Test that empty utterances are filtered out."""
        records = SAMPLE_TRANSCRIPTS.copy()
        records.append({
            "transcript_id": "T006", "customer_utterance": "",
            "intent": "refund_request", "channel": "live_chat",
            "sentiment": "neutral", "complexity": "simple"
        })
        path = _create_temp_csv(records)
        try:
            loader = DatasetLoader()
            loaded = loader.load(path)
            assert len(loaded) == 5, f"Expected 5 (empty filtered), got {len(loaded)}"
            print("  PASS: test_empty_text_filtered")
        finally:
            os.unlink(path)
    
    def test_file_not_found(self):
        """Test FileNotFoundError for missing files."""
        loader = DatasetLoader()
        try:
            loader.load("nonexistent_file.csv")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            print("  PASS: test_file_not_found")
    
    def test_preprocess_text(self):
        """Test text preprocessing (whitespace normalization)."""
        loader = DatasetLoader()
        result = loader.preprocess_text("  hello   world  ")
        assert result == "hello world", f"Expected 'hello world', got '{result}'"
        print("  PASS: test_preprocess_text")


# =============================================================================
# Paraphraser Tests (Rule-Based)
# =============================================================================

class TestRuleBasedParaphraser:
    
    def test_generates_variants(self):
        """Test that paraphraser generates non-empty variants."""
        rp = RuleBasedParaphraser(seed=42)
        variants = rp.paraphrase("I need a refund for my damaged order", num_variants=3)
        assert len(variants) > 0, "Expected at least 1 variant"
        print(f"  PASS: test_generates_variants ({len(variants)} variants)")
    
    def test_variants_differ_from_original(self):
        """Test that variants are different from the original."""
        rp = RuleBasedParaphraser(seed=42)
        original = "I need a refund for my damaged order"
        variants = rp.paraphrase(original, num_variants=3)
        for v in variants:
            assert v.lower() != original.lower(), f"Variant identical to original: {v}"
        print("  PASS: test_variants_differ_from_original")
    
    def test_seed_reproducibility(self):
        """Test that the same seed produces the same results."""
        rp1 = RuleBasedParaphraser(seed=100)
        rp2 = RuleBasedParaphraser(seed=100)
        v1 = rp1.paraphrase("Cancel my order please", num_variants=2)
        v2 = rp2.paraphrase("Cancel my order please", num_variants=2)
        assert v1 == v2, "Same seed should produce identical results"
        print("  PASS: test_seed_reproducibility")
    
    def test_config(self):
        """Test configuration retrieval."""
        rp = RuleBasedParaphraser(seed=42)
        config = rp.get_config()
        assert config["type"] == "rule_based"
        assert config["seed"] == 42
        print("  PASS: test_config")


# =============================================================================
# NoiseGenerator Tests
# =============================================================================

class TestNoiseGenerator:
    
    def test_keyboard_noise(self):
        """Test keyboard noise produces modified output."""
        ng = NoiseGenerator(seed=42)
        original = "I need a refund for my damaged order"
        # Run multiple times since noise is probabilistic
        changed = False
        for _ in range(10):
            result = ng.apply_keyboard_noise(original)
            if result != original:
                changed = True
                break
        assert changed, "Keyboard noise should modify text at least once in 10 tries"
        print("  PASS: test_keyboard_noise")
    
    def test_linguistic_noise(self):
        """Test linguistic noise applies domain synonym swaps."""
        ng = NoiseGenerator(seed=42)
        original = "I need a refund for my damaged order"
        result = ng.apply_linguistic_noise(original)
        # Should produce some variant (synonym swap, contraction toggle, or case change)
        assert isinstance(result, str) and len(result) > 0
        print(f"  PASS: test_linguistic_noise -> '{result}'")
    
    def test_char_noise(self):
        """Test character-level noise mutations."""
        ng = NoiseGenerator(seed=42)
        original = "My payment was declined at checkout"
        changed = False
        for _ in range(10):
            result = ng.apply_char_noise(original)
            if result != original:
                changed = True
                break
        assert changed, "Char noise should modify text"
        print("  PASS: test_char_noise")
    
    def test_combined_noise(self):
        """Test combined noise strategy."""
        ng = NoiseGenerator(seed=42)
        original = "Where is my package tracking number"
        result = ng.apply_combined_noise(original)
        assert isinstance(result, str)
        print(f"  PASS: test_combined_noise -> '{result}'")
    
    def test_budget_enforcement(self):
        """Test that noise budget is enforced."""
        ng = NoiseGenerator(seed=42, max_char_perturbation_rate=0.01)
        original = "refund"
        # With very tight budget, heavily modified text should be reverted
        heavily_modified = "xxxxxx"
        result = ng._enforce_budget(original, heavily_modified)
        assert result == original, "Budget enforcement should revert over-modified text"
        print("  PASS: test_budget_enforcement")
    
    def test_number_preservation(self):
        """Test that numeric tokens are preserved."""
        ng = NoiseGenerator(seed=42, preserve_numbers=True)
        # Numbers should not be modified by manual keyboard noise
        original = "order 12345 was placed"
        result = ng._manual_keyboard_noise(original)
        # Check that "12345" is still present
        assert "12345" in result, f"Number '12345' should be preserved, got: {result}"
        print("  PASS: test_number_preservation")
    
    def test_config(self):
        """Test noise generator configuration."""
        ng = NoiseGenerator(seed=42)
        config = ng.get_config()
        assert config["seed"] == 42
        assert config["preserve_numbers"] == True
        print("  PASS: test_config")


# =============================================================================
# SemanticValidator Tests
# =============================================================================

class TestSemanticValidator:
    
    def test_identical_text_high_similarity(self):
        """Test that identical texts have high similarity."""
        v = SemanticValidator()
        score = v.compute_similarity(
            "I need a refund",
            "I need a refund"
        )
        assert score >= 0.95, f"Identical text should have sim >= 0.95, got {score}"
        print(f"  PASS: test_identical_text (similarity={score})")
    
    def test_similar_text_moderate_similarity(self):
        """Test that semantically similar texts have positive similarity."""
        v = SemanticValidator()
        score = v.compute_similarity(
            "I need a refund for my order",
            "Please give me my money back for the order"
        )
        assert score > 0.05, f"Similar texts should have positive sim, got {score}"
        print(f"  PASS: test_similar_text (similarity={score})")
    
    def test_dissimilar_text_low_similarity(self):
        """Test that unrelated texts have low similarity."""
        v = SemanticValidator()
        score = v.compute_similarity(
            "I need a refund for my order",
            "The weather is sunny and warm today"
        )
        assert score < 0.5, f"Dissimilar texts should have low sim, got {score}"
        print(f"  PASS: test_dissimilar_text (similarity={score})")
    
    def test_validation_window(self):
        """Test the validation acceptance window."""
        v = SemanticValidator(similarity_floor=0.5, similarity_ceiling=0.95)
        assert v.is_valid(0.7) == True
        assert v.is_valid(0.3) == False   # Below floor
        assert v.is_valid(0.99) == False  # Above ceiling
        print("  PASS: test_validation_window")
    
    def test_validate_variant_detail(self):
        """Test detailed variant validation results."""
        v = SemanticValidator()
        result = v.validate_variant(
            "I need a refund",
            "Please reimburse my purchase"
        )
        assert "similarity" in result
        assert "is_valid" in result
        assert "rejection_reason" in result
        assert isinstance(result["similarity"], float)
        print(f"  PASS: test_validate_variant_detail (sim={result['similarity']})")
    
    def test_filter_variants(self):
        """Test batch filtering of variants."""
        v = SemanticValidator(similarity_floor=0.3, similarity_ceiling=0.99)
        accepted, rejected = v.filter_variants(
            "I need a refund",
            ["Please reimburse me", "The weather is nice", "I need a refund"]
        )
        assert isinstance(accepted, list)
        assert isinstance(rejected, list)
        total = len(accepted) + len(rejected)
        assert total == 3, f"Expected 3 total results, got {total}"
        print(f"  PASS: test_filter_variants (accepted={len(accepted)}, rejected={len(rejected)})")


# =============================================================================
# Integration Test
# =============================================================================

class TestPipelineIntegration:
    
    def test_end_to_end(self):
        """Test full pipeline execution with sample data."""
        from pipeline import AugmentationPipeline
        
        # Create temp CSV
        path = _create_temp_csv(SAMPLE_TRANSCRIPTS)
        
        try:
            pipeline = AugmentationPipeline(
                num_variants=2,
                strategies=["keyboard_noise", "linguistic_noise"],
                use_transformer_paraphraser=False,
                seed=42,
                enable_validation=True
            )
            
            results = pipeline.run(input_path=path)
            
            assert len(results) > 0, "Pipeline should produce augmented records"
            
            # Check record structure
            first = results[0]
            assert "original_text" in first
            assert "augmented_text" in first
            assert "intent" in first
            assert "augmentation_strategy" in first
            assert "similarity_score" in first
            
            # Check stats
            assert pipeline.stats["total_originals"] == 5
            assert pipeline.stats["total_generated"] > 0
            
            print(f"  PASS: test_end_to_end ({len(results)} augmented records)")
            
            # Test export
            fd, out_path = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            try:
                pipeline.export(results, output_path=out_path)
                # Verify exported file
                with open(out_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    exported = list(reader)
                assert len(exported) == len(results)
                print(f"  PASS: test_export ({len(exported)} rows exported)")
            finally:
                os.unlink(out_path)
            
        finally:
            os.unlink(path)


# =============================================================================
# Test Runner
# =============================================================================

def run_all_tests():
    """Run all test suites."""
    print("=" * 60)
    print("  IntentAugment AI - Pipeline Test Suite")
    print("=" * 60)
    
    test_classes = [
        ("DatasetLoader", TestDatasetLoader),
        ("RuleBasedParaphraser", TestRuleBasedParaphraser),
        ("NoiseGenerator", TestNoiseGenerator),
        ("SemanticValidator", TestSemanticValidator),
        ("Pipeline Integration", TestPipelineIntegration),
    ]
    
    total_passed = 0
    total_failed = 0
    
    for name, cls in test_classes:
        print(f"\n--- {name} Tests ---")
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        
        for method_name in methods:
            try:
                getattr(instance, method_name)()
                total_passed += 1
            except Exception as e:
                print(f"  FAIL: {method_name} -> {e}")
                total_failed += 1
    
    print("\n" + "=" * 60)
    print(f"  Results: {total_passed} passed, {total_failed} failed")
    print("=" * 60)
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
