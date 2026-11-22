"""
IntentAugment AI - Transformer-Based Paraphraser Module
========================================================
Implements controlled contextual paraphrasing using Hugging Face
sequence-to-sequence models (T5 / Pegasus) for generating diverse
customer support transcript variations while preserving intent.

Author: Abhishek Manek
Deliverable: #2 (Week 2) - Transcript Paraphrasing & Noise Augmentation Pipeline
"""

import os
import random
from typing import List, Optional, Dict, Any

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[Paraphraser] Warning: transformers/torch not installed. "
          "Install with: pip install transformers torch")


class TranscriptParaphraser:
    """
    Generates contextual paraphrases of customer support transcripts using
    Hugging Face T5-based sequence-to-sequence models.
    
    Supports:
        - Beam search decoding with diversity penalty
        - Nucleus (top-p) and top-k sampling
        - Temperature-controlled diversity
        - Configurable repetition penalty
        - Batch paraphrase generation
    
    Example usage:
        >>> paraphraser = TranscriptParaphraser(model_name="Vamsi/T5_Paraphrase_Paws")
        >>> results = paraphraser.paraphrase("I need a refund for my order")
        >>> print(results)
        ['I want my money back for the purchase', 'Please refund my order', ...]
    """
    
    # Prefix required by T5-based paraphrase models
    T5_PARAPHRASE_PREFIX = "paraphrase: "
    
    def __init__(
        self,
        model_name: str = "Vamsi/T5_Paraphrase_Paws",
        max_input_length: int = 128,
        max_output_length: int = 128,
        num_return_sequences: int = 3,
        num_beams: int = 5,
        temperature: float = 1.5,
        top_p: float = 0.92,
        top_k: int = 50,
        do_sample: bool = True,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        diversity_penalty: float = 0.0,
        device: str = "auto",
        seed: int = 42
    ):
        """
        Initialize the TranscriptParaphraser.
        
        Args:
            model_name: Hugging Face model identifier for paraphrasing.
            max_input_length: Maximum token length for input text.
            max_output_length: Maximum token length for generated output.
            num_return_sequences: Number of paraphrases to generate per input.
            num_beams: Beam search width.
            temperature: Sampling temperature (higher = more diverse).
            top_p: Nucleus sampling cumulative probability threshold.
            top_k: Top-k sampling number of candidates.
            do_sample: Whether to use sampling (True) or greedy/beam (False).
            repetition_penalty: Penalty for repeating tokens.
            length_penalty: Length normalization factor.
            diversity_penalty: Penalty for similar beam groups.
            device: Device selection ('auto', 'cpu', 'cuda', 'mps').
            seed: Random seed for reproducibility.
        """
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_return_sequences = num_return_sequences
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.do_sample = do_sample
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty
        self.diversity_penalty = diversity_penalty
        self.seed = seed
        
        self._model = None
        self._tokenizer = None
        self._device = None
        
        if HF_AVAILABLE:
            self._resolve_device(device)
            random.seed(seed)
            torch.manual_seed(seed)
        
        self._is_loaded = False
    
    def _resolve_device(self, device: str):
        """Resolve the compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)
        print(f"[Paraphraser] Using device: {self._device}")
    
    def load_model(self):
        """
        Load the paraphrase model and tokenizer from Hugging Face.
        Downloads the model on first use and caches it locally.
        """
        if not HF_AVAILABLE:
            raise RuntimeError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )
        
        if self._is_loaded:
            print(f"[Paraphraser] Model already loaded: {self.model_name}")
            return
        
        print(f"[Paraphraser] Loading model: {self.model_name}...")
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model = self._model.to(self._device)
            self._model.eval()
            self._is_loaded = True
            print(f"[Paraphraser] Model loaded successfully on {self._device}")
        except Exception as e:
            print(f"[Paraphraser] Error loading model '{self.model_name}': {e}")
            raise
    
    def paraphrase(self, text: str, num_variants: Optional[int] = None) -> List[str]:
        """
        Generate paraphrase variants of a customer support transcript.
        
        Args:
            text: Original customer utterance to paraphrase.
            num_variants: Number of variants to generate (overrides default).
            
        Returns:
            List of paraphrased text strings.
        """
        if not self._is_loaded:
            self.load_model()
        
        n = num_variants or self.num_return_sequences
        
        # Add T5 paraphrase task prefix
        input_text = self.T5_PARAPHRASE_PREFIX + text
        
        # Tokenize input
        encoding = self._tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(self._device)
        
        # Generate paraphrases
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                max_length=self.max_output_length,
                num_return_sequences=n,
                num_beams=max(self.num_beams, n),  # Beams must be >= sequences
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                do_sample=self.do_sample,
                repetition_penalty=self.repetition_penalty,
                length_penalty=self.length_penalty,
                early_stopping=True
            )
        
        # Decode and deduplicate results
        paraphrases = []
        seen = set()
        original_lower = text.strip().lower()
        
        for output in outputs:
            decoded = self._tokenizer.decode(output, skip_special_tokens=True).strip()
            decoded_lower = decoded.lower()
            
            # Skip exact duplicates and identity copies
            if decoded_lower not in seen and decoded_lower != original_lower:
                paraphrases.append(decoded)
                seen.add(decoded_lower)
        
        return paraphrases[:n]
    
    def paraphrase_batch(
        self, texts: List[str], num_variants: Optional[int] = None
    ) -> List[List[str]]:
        """
        Generate paraphrases for a batch of transcripts.
        
        Args:
            texts: List of original customer utterances.
            num_variants: Number of variants per utterance.
            
        Returns:
            List of lists, each containing paraphrased variants.
        """
        results = []
        for i, text in enumerate(texts):
            paraphrases = self.paraphrase(text, num_variants)
            results.append(paraphrases)
            if (i + 1) % 25 == 0:
                print(f"[Paraphraser] Processed {i + 1}/{len(texts)} transcripts...")
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Return current paraphraser configuration as a dictionary."""
        return {
            "model_name": self.model_name,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "num_return_sequences": self.num_return_sequences,
            "num_beams": self.num_beams,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "repetition_penalty": self.repetition_penalty,
            "device": str(self._device) if self._device else "not_set"
        }


# =============================================================================
# Lightweight fallback paraphraser (no model download required)
# Uses rule-based synonym substitution and sentence restructuring
# =============================================================================

class RuleBasedParaphraser:
    """
    A lightweight rule-based paraphraser that generates variations
    without requiring Hugging Face model downloads.
    
    Uses synonym dictionaries, sentence templates, and structural
    transformations common in customer support contexts.
    
    This serves as a fallback when transformer models are unavailable
    and also as a complementary augmentation strategy.
    """
    
    # Domain-specific synonym clusters for customer support vocabulary
    SYNONYM_MAP = {
        # Refund domain
        "refund": ["reimbursement", "money back", "credit", "repayment"],
        "return": ["send back", "give back", "ship back"],
        "damaged": ["broken", "defective", "faulty", "busted"],
        "wrong": ["incorrect", "not what I ordered", "different from"],
        
        # Cancel domain
        "cancel": ["terminate", "revoke", "abort", "void", "stop"],
        "order": ["purchase", "transaction", "request"],
        "immediately": ["right now", "asap", "right away", "urgently"],
        
        # Shipping domain
        "shipping": ["delivery", "shipment", "dispatch"],
        "tracking": ["trace", "locate", "follow"],
        "package": ["parcel", "item", "shipment"],
        "delivered": ["arrived", "received", "dropped off"],
        "late": ["delayed", "overdue", "behind schedule"],
        
        # Payment domain
        "payment": ["transaction", "charge", "billing"],
        "charged": ["billed", "debited", "deducted"],
        "declined": ["rejected", "failed", "not accepted"],
        "card": ["credit card", "debit card", "payment card"],
        
        # Account domain
        "password": ["passcode", "login credentials", "access code"],
        "locked": ["blocked", "frozen", "suspended"],
        "login": ["sign in", "log on", "access"],
        "account": ["profile", "user account"],
        
        # Technical domain
        "error": ["issue", "problem", "bug", "glitch"],
        "crash": ["freeze", "stop working", "fail"],
        "update": ["upgrade", "patch", "new version"],
        "working": ["functioning", "operating", "running"],
        
        # Subscription domain
        "subscription": ["membership", "plan", "service"],
        "renew": ["extend", "continue", "prolong"],
        "upgrade": ["switch to higher plan", "move to premium"],
        
        # General
        "help": ["assist", "support", "guide"],
        "please": ["kindly", ""],
        "need": ["require", "want", "must have"],
        "get": ["receive", "obtain", "acquire"],
        "check": ["verify", "look into", "examine", "review"],
    }
    
    # Contraction expansion/contraction pairs
    CONTRACTIONS = {
        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
        "didn't": "did not",
        "doesn't": "does not",
        "isn't": "is not",
        "wasn't": "was not",
        "aren't": "are not",
        "weren't": "were not",
        "haven't": "have not",
        "hasn't": "has not",
        "hadn't": "had not",
        "shouldn't": "should not",
        "wouldn't": "would not",
        "couldn't": "could not",
        "i'm": "I am",
        "i've": "I have",
        "i'll": "I will",
        "i'd": "I would",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is",
        "they're": "they are",
        "we're": "we are",
        "you're": "you are",
        "let's": "let us",
    }
    
    def __init__(self, seed: int = 42):
        """Initialize with a random seed for reproducibility."""
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Build reverse contraction map
        self.expansion_map = dict(self.CONTRACTIONS)
        self.contraction_map = {v.lower(): k for k, v in self.CONTRACTIONS.items()}
    
    def paraphrase(self, text: str, num_variants: int = 3) -> List[str]:
        """
        Generate paraphrase variants using rule-based transformations.
        
        Strategies applied:
            1. Synonym substitution from domain-specific vocabulary
            2. Contraction expansion / contraction toggling
            3. Politeness prefix/suffix modulation
            4. Active/passive restructuring hints
        
        Args:
            text: Original customer utterance.
            num_variants: Number of variants to generate.
            
        Returns:
            List of paraphrased text variants.
        """
        variants = set()
        attempts = 0
        max_attempts = num_variants * 5
        
        while len(variants) < num_variants and attempts < max_attempts:
            attempts += 1
            strategy = self.rng.choice([
                self._synonym_swap,
                self._toggle_contractions,
                self._modulate_politeness,
                self._combined_transform
            ])
            variant = strategy(text)
            if variant.lower().strip() != text.lower().strip():
                variants.add(variant)
        
        return list(variants)[:num_variants]
    
    def _synonym_swap(self, text: str) -> str:
        """Replace one or two words with domain-specific synonyms."""
        words = text.split()
        modified = words.copy()
        
        indices_to_try = list(range(len(words)))
        self.rng.shuffle(indices_to_try)
        swaps_done = 0
        
        for idx in indices_to_try:
            word_lower = words[idx].strip(".,!?;:").lower()
            if word_lower in self.SYNONYM_MAP:
                synonyms = self.SYNONYM_MAP[word_lower]
                synonym = self.rng.choice(synonyms)
                if synonym:  # Skip empty string synonyms
                    # Preserve original punctuation
                    trailing = ""
                    if words[idx] and words[idx][-1] in ".,!?;:":
                        trailing = words[idx][-1]
                    modified[idx] = synonym + trailing
                    swaps_done += 1
                    if swaps_done >= 2:
                        break
        
        return " ".join(modified)
    
    def _toggle_contractions(self, text: str) -> str:
        """Expand contractions or contract expanded forms."""
        result = text
        
        # Try expanding contractions first
        for contraction, expansion in self.expansion_map.items():
            if contraction in result.lower():
                # Case-insensitive replacement
                import re
                pattern = re.compile(re.escape(contraction), re.IGNORECASE)
                result = pattern.sub(expansion, result, count=1)
                return result
        
        # Try contracting expanded forms
        for expansion, contraction in self.contraction_map.items():
            if expansion in result.lower():
                import re
                pattern = re.compile(re.escape(expansion), re.IGNORECASE)
                result = pattern.sub(contraction, result, count=1)
                return result
        
        return result
    
    def _modulate_politeness(self, text: str) -> str:
        """Add or remove politeness markers."""
        polite_prefixes = [
            "Could you please ", "I would appreciate if you could ",
            "Would you mind ", "Hi, ", "Hello, ",
            "Excuse me, ", "Sorry but "
        ]
        polite_suffixes = [
            " please", " thanks", " thank you", " if possible",
            " when you get a chance", " at your earliest convenience"
        ]
        
        result = text
        
        # Remove existing politeness markers
        for prefix in polite_prefixes:
            if result.lower().startswith(prefix.lower()):
                result = result[len(prefix):]
                # Capitalize first letter
                if result:
                    result = result[0].upper() + result[1:]
                return result
        
        for suffix in polite_suffixes:
            if result.lower().rstrip(".,!?").endswith(suffix.strip().lower()):
                end_pos = result.lower().rfind(suffix.strip().lower())
                if end_pos > 0:
                    result = result[:end_pos].rstrip()
                    return result
        
        # Add a politeness marker
        if self.rng.random() > 0.5:
            prefix = self.rng.choice(polite_prefixes)
            result = prefix + result[0].lower() + result[1:]
        else:
            suffix = self.rng.choice(polite_suffixes)
            result = result.rstrip(".,!?") + suffix + "."
        
        return result
    
    def _combined_transform(self, text: str) -> str:
        """Apply multiple light transformations together."""
        result = self._synonym_swap(text)
        if self.rng.random() > 0.5:
            result = self._toggle_contractions(result)
        return result
    
    def get_config(self) -> Dict[str, Any]:
        """Return current configuration."""
        return {
            "type": "rule_based",
            "synonym_map_size": len(self.SYNONYM_MAP),
            "contraction_pairs": len(self.CONTRACTIONS),
            "seed": self.seed
        }


if __name__ == "__main__":
    # Demo with rule-based paraphraser (no model download needed)
    print("=" * 60)
    print("  IntentAugment AI - Paraphraser Module Demo")
    print("=" * 60)
    
    rp = RuleBasedParaphraser(seed=42)
    
    test_sentences = [
        "I need a refund for my damaged order",
        "Can you cancel my order immediately please?",
        "Where is my package? It hasn't arrived yet",
        "My payment was declined at checkout",
        "I can't login to my account, password reset isn't working",
    ]
    
    for sentence in test_sentences:
        print(f"\nOriginal : {sentence}")
        variants = rp.paraphrase(sentence, num_variants=3)
        for i, v in enumerate(variants, 1):
            print(f"  Variant {i}: {v}")
    
    print("\n" + "=" * 60)
    
    # If transformer models are available, demo those too
    if HF_AVAILABLE:
        print("\n[Demo] Transformer paraphraser available.")
        print("[Demo] To use: paraphraser = TranscriptParaphraser()")
        print("[Demo] paraphraser.load_model()  # Downloads model on first run")
        print("[Demo] paraphraser.paraphrase('I need a refund')")
    else:
        print("\n[Demo] Install transformers+torch for transformer-based paraphrasing:")
        print("  pip install transformers torch")
