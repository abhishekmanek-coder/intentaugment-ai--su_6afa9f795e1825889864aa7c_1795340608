"""
IntentAugment AI - Semantic Validator Module
==============================================
Validates augmented transcripts for intent preservation using
Sentence-Transformers dense embeddings and cosine similarity.

Ensures augmented text maintains semantic alignment with the original
customer utterance while rejecting trivial identity copies and
significant semantic drift.

Author: Abhishek Manek
Deliverable: #2 (Week 2) - Transcript Paraphrasing & Noise Augmentation Pipeline
"""

import math
from typing import List, Dict, Tuple, Optional, Any

try:
    from sentence_transformers import SentenceTransformer, util
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    print("[Validator] Warning: sentence-transformers not installed. "
          "Install with: pip install sentence-transformers")


class SemanticValidator:
    """
    Validates augmented transcripts against their originals using
    dense sentence embeddings and cosine similarity scoring.
    
    Accepts augmented variants that fall within the configured
    similarity window [floor, ceiling]:
        - Below floor (< 0.80): Semantic drift detected, variant rejected.
        - Above ceiling (> 0.98): Trivial copy detected, variant rejected.
        - Within window: Variant accepted as valid augmentation.
    
    Example usage:
        >>> validator = SemanticValidator()
        >>> score = validator.compute_similarity("I need a refund", "Give me my money back")
        >>> print(f"Similarity: {score:.4f}")
        >>> is_valid = validator.is_valid(score)
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        similarity_floor: float = 0.80,
        similarity_ceiling: float = 0.98,
        batch_size: int = 32,
        device: str = "cpu"
    ):
        """
        Initialize the SemanticValidator.
        
        Args:
            model_name: Sentence-Transformers model identifier.
            similarity_floor: Minimum cosine similarity to accept (default: 0.80).
            similarity_ceiling: Maximum cosine similarity to accept (default: 0.98).
            batch_size: Batch size for embedding computation.
            device: Compute device ('cpu', 'cuda', 'mps').
        """
        self.model_name = model_name
        self.similarity_floor = similarity_floor
        self.similarity_ceiling = similarity_ceiling
        self.batch_size = batch_size
        self.device = device
        
        self._model = None
        self._is_loaded = False
    
    def load_model(self):
        """Load the Sentence-Transformers embedding model."""
        if not SBERT_AVAILABLE:
            raise RuntimeError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        if self._is_loaded:
            return
        
        print(f"[Validator] Loading embedding model: {self.model_name}...")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._is_loaded = True
        print(f"[Validator] Model loaded successfully on {self.device}")
    
    def compute_similarity(self, original: str, augmented: str) -> float:
        """
        Compute cosine similarity between original and augmented text.
        
        Args:
            original: Original customer utterance.
            augmented: Augmented text variant.
            
        Returns:
            Cosine similarity score in range [-1, 1].
        """
        if self._model is not None:
            embeddings = self._model.encode(
                [original, augmented],
                convert_to_tensor=True,
                batch_size=self.batch_size
            )
            similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
            return round(similarity, 4)
        else:
            # Fallback: simple word overlap similarity (Jaccard-like)
            return self._fallback_similarity(original, augmented)
    
    def compute_similarity_batch(
        self,
        originals: List[str],
        augmented_list: List[str]
    ) -> List[float]:
        """
        Compute cosine similarities for a batch of original-augmented pairs.
        
        Args:
            originals: List of original texts.
            augmented_list: List of augmented texts (same length).
            
        Returns:
            List of cosine similarity scores.
        """
        assert len(originals) == len(augmented_list), \
            "Original and augmented lists must have the same length."
        
        if self._model is not None:
            orig_embeddings = self._model.encode(
                originals, convert_to_tensor=True, batch_size=self.batch_size
            )
            aug_embeddings = self._model.encode(
                augmented_list, convert_to_tensor=True, batch_size=self.batch_size
            )
            similarities = util.cos_sim(orig_embeddings, aug_embeddings)
            return [round(similarities[i][i].item(), 4) for i in range(len(originals))]
        else:
            return [
                self._fallback_similarity(o, a)
                for o, a in zip(originals, augmented_list)
            ]
    
    def is_valid(self, similarity_score: float) -> bool:
        """
        Check if an augmented variant passes the quality gate.
        
        Args:
            similarity_score: Cosine similarity between original and variant.
            
        Returns:
            True if the variant is within the acceptable similarity window.
        """
        return self.similarity_floor <= similarity_score <= self.similarity_ceiling
    
    def validate_variant(
        self, original: str, augmented: str
    ) -> Dict[str, Any]:
        """
        Validate a single augmented variant and return detailed results.
        
        Args:
            original: Original customer utterance.
            augmented: Augmented text variant.
            
        Returns:
            Dictionary with validation results:
                - similarity: float score
                - is_valid: bool
                - rejection_reason: str or None
        """
        similarity = self.compute_similarity(original, augmented)
        valid = self.is_valid(similarity)
        
        reason = None
        if not valid:
            if similarity < self.similarity_floor:
                reason = f"semantic_drift (sim={similarity:.4f} < floor={self.similarity_floor})"
            elif similarity > self.similarity_ceiling:
                reason = f"trivial_copy (sim={similarity:.4f} > ceiling={self.similarity_ceiling})"
        
        return {
            "original": original,
            "augmented": augmented,
            "similarity": similarity,
            "is_valid": valid,
            "rejection_reason": reason
        }
    
    def filter_variants(
        self, original: str, variants: List[str]
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter a list of augmented variants into accepted and rejected.
        
        Args:
            original: Original customer utterance.
            variants: List of augmented text variants.
            
        Returns:
            Tuple of (accepted_list, rejected_list), each containing
            validation result dictionaries.
        """
        accepted = []
        rejected = []
        
        for variant in variants:
            result = self.validate_variant(original, variant)
            if result["is_valid"]:
                accepted.append(result)
            else:
                rejected.append(result)
        
        return accepted, rejected
    
    def _fallback_similarity(self, text1: str, text2: str) -> float:
        """
        Fallback word-overlap similarity when Sentence-Transformers
        is not available. Uses weighted Jaccard + character overlap.
        
        Args:
            text1: First text string.
            text2: Second text string.
            
        Returns:
            Approximate similarity score in range [0, 1].
        """
        # Word-level Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Character n-gram overlap (bigrams)
        def char_ngrams(text, n=2):
            text = text.lower().replace(" ", "")
            return set(text[i:i+n] for i in range(len(text) - n + 1))
        
        ngrams1 = char_ngrams(text1)
        ngrams2 = char_ngrams(text2)
        
        if ngrams1 and ngrams2:
            ngram_overlap = len(ngrams1 & ngrams2) / len(ngrams1 | ngrams2)
        else:
            ngram_overlap = 0.0
        
        # Weighted combination
        combined = 0.6 * jaccard + 0.4 * ngram_overlap
        return round(combined, 4)
    
    def get_config(self) -> Dict[str, Any]:
        """Return current validator configuration."""
        return {
            "model_name": self.model_name,
            "similarity_floor": self.similarity_floor,
            "similarity_ceiling": self.similarity_ceiling,
            "batch_size": self.batch_size,
            "device": self.device,
            "model_loaded": self._is_loaded,
            "sbert_available": SBERT_AVAILABLE
        }


if __name__ == "__main__":
    print("=" * 60)
    print("  IntentAugment AI - Semantic Validator Module Demo")
    print("=" * 60)
    
    validator = SemanticValidator()
    
    # Try loading the model
    if SBERT_AVAILABLE:
        try:
            validator.load_model()
        except Exception as e:
            print(f"[Demo] Model load failed: {e}")
            print("[Demo] Using fallback word-overlap similarity.")
    else:
        print("[Demo] Using fallback word-overlap similarity.")
    
    # Test pairs
    test_pairs = [
        ("I need a refund for my damaged order",
         "Please give me my money back for the broken item"),
        ("I need a refund for my damaged order",
         "The weather is beautiful today"),
        ("I need a refund for my damaged order",
         "I need a refund for my damaged order"),
        ("Can you cancel my order immediately?",
         "Please terminate my purchase right now"),
        ("Where is my package?",
         "I want to track my parcel location"),
    ]
    
    print(f"\nSimilarity Window: [{validator.similarity_floor}, {validator.similarity_ceiling}]")
    print("-" * 60)
    
    for original, augmented in test_pairs:
        result = validator.validate_variant(original, augmented)
        status = "ACCEPTED" if result["is_valid"] else "REJECTED"
        reason = f" ({result['rejection_reason']})" if result["rejection_reason"] else ""
        print(f"\nOriginal  : {original}")
        print(f"Augmented : {augmented}")
        print(f"Similarity: {result['similarity']:.4f}  -->  {status}{reason}")
    
    print("\n" + "=" * 60)
