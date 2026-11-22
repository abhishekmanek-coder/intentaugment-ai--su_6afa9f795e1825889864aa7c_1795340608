"""
IntentAugment AI - Noise & Typo Generator Module
==================================================
Implements controlled noise injection for customer support transcripts
using NLPAug (keyboard typos, OCR errors, character mutations) and
spaCy-based linguistic perturbations (synonym substitution, contraction
toggling, case variation).

Author: Abhishek Manek
Deliverable: #2 (Week 2) - Transcript Paraphrasing & Noise Augmentation Pipeline
"""

import random
import re
import string
from typing import List, Dict, Any, Optional, Set

# Conditional imports for NLPAug
try:
    import nlpaug.augmenter.char as nac
    import nlpaug.augmenter.word as naw
    NLPAUG_AVAILABLE = True
except ImportError:
    NLPAUG_AVAILABLE = False
    print("[NoiseGenerator] Warning: nlpaug not installed. "
          "Install with: pip install nlpaug")

# Conditional imports for spaCy
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("[NoiseGenerator] Warning: spaCy not installed. "
          "Install with: pip install spacy && python -m spacy download en_core_web_sm")


class NoiseGenerator:
    """
    Generates realistic noise and typo variations in customer support
    transcripts to simulate real-world input imperfections.
    
    Noise Types:
        1. Keyboard Noise: QWERTY adjacent-key substitutions simulating
           fat-finger typing errors on mobile and desktop keyboards.
        2. OCR Noise: Character substitutions mimicking optical character
           recognition and handwriting transcription errors.
        3. Character Noise: Random character insertions, deletions,
           swaps, and substitutions.
        4. Linguistic Noise: spaCy-powered synonym replacement,
           contraction toggling, and case variation.
    
    Example usage:
        >>> noise_gen = NoiseGenerator(seed=42)
        >>> result = noise_gen.apply_keyboard_noise("I need a refund")
        >>> print(result)  # e.g., "I nedd a refubd"
    """
    
    # QWERTY keyboard adjacency map for realistic typo simulation
    QWERTY_NEIGHBORS = {
        'q': 'wa', 'w': 'qeas', 'e': 'wrds', 'r': 'etfs', 't': 'rygs',
        'y': 'tuhs', 'u': 'yijs', 'i': 'uoks', 'o': 'ipls', 'p': 'ol',
        'a': 'qwsz', 's': 'awedxz', 'd': 'serfcx', 'f': 'drtgvc',
        'g': 'ftyhbv', 'h': 'gyujnb', 'j': 'huikmn', 'k': 'jiolm',
        'l': 'kop', 'z': 'asx', 'x': 'zsdc', 'c': 'xdfv',
        'v': 'cfgb', 'b': 'vghn', 'n': 'bhjm', 'm': 'njk',
    }
    
    # Common contraction pairs for toggling
    CONTRACTIONS = {
        "can't": "cannot", "won't": "will not", "don't": "do not",
        "didn't": "did not", "doesn't": "does not", "isn't": "is not",
        "wasn't": "was not", "aren't": "are not", "haven't": "have not",
        "hasn't": "has not", "couldn't": "could not", "shouldn't": "should not",
        "wouldn't": "would not", "i'm": "i am", "i've": "i have",
        "i'll": "i will", "i'd": "i would", "it's": "it is",
        "that's": "that is", "there's": "there is", "they're": "they are",
        "we're": "we are", "you're": "you are",
    }
    
    # Domain-aware word-level synonyms for linguistic noise
    DOMAIN_SYNONYMS = {
        "refund": ["reimbursement", "money back", "credit back"],
        "cancel": ["terminate", "revoke", "abort", "void"],
        "order": ["purchase", "transaction"],
        "package": ["parcel", "shipment", "item"],
        "payment": ["transaction", "charge", "billing"],
        "account": ["profile", "user account"],
        "error": ["issue", "problem", "bug", "glitch"],
        "help": ["assist", "support"],
        "broken": ["damaged", "defective", "faulty"],
        "working": ["functioning", "operating"],
        "update": ["upgrade", "patch"],
        "login": ["sign in", "log on"],
        "password": ["passcode", "credentials"],
        "shipping": ["delivery", "dispatch"],
        "tracking": ["tracing", "locating"],
        "subscription": ["membership", "plan"],
    }
    
    def __init__(
        self,
        seed: int = 42,
        keyboard_aug_char_p: float = 0.08,
        keyboard_aug_word_p: float = 0.15,
        ocr_aug_char_p: float = 0.06,
        char_aug_p: float = 0.05,
        synonym_p: float = 0.15,
        max_char_perturbation_rate: float = 0.12,
        max_word_perturbation_rate: float = 0.15,
        preserve_numbers: bool = True,
        min_char: int = 3
    ):
        """
        Initialize the NoiseGenerator.
        
        Args:
            seed: Random seed for reproducibility.
            keyboard_aug_char_p: Per-character probability for keyboard noise.
            keyboard_aug_word_p: Per-word probability for keyboard noise.
            ocr_aug_char_p: Per-character probability for OCR noise.
            char_aug_p: Per-character probability for random char noise.
            synonym_p: Probability of replacing a word with a domain synonym.
            max_char_perturbation_rate: Maximum fraction of characters to modify.
            max_word_perturbation_rate: Maximum fraction of words to modify.
            preserve_numbers: Whether to preserve numeric tokens.
            min_char: Minimum word length to apply character-level noise.
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.keyboard_aug_char_p = keyboard_aug_char_p
        self.keyboard_aug_word_p = keyboard_aug_word_p
        self.ocr_aug_char_p = ocr_aug_char_p
        self.char_aug_p = char_aug_p
        self.synonym_p = synonym_p
        self.max_char_perturbation_rate = max_char_perturbation_rate
        self.max_word_perturbation_rate = max_word_perturbation_rate
        self.preserve_numbers = preserve_numbers
        self.min_char = min_char
        
        # Initialize NLPAug augmenters if available
        self._keyboard_aug = None
        self._ocr_aug = None
        self._char_aug = None
        self._spacy_nlp = None
        
        self._init_augmenters()
    
    def _init_augmenters(self):
        """Initialize NLPAug and spaCy augmenters."""
        if NLPAUG_AVAILABLE:
            try:
                self._keyboard_aug = nac.KeyboardAug(
                    aug_char_p=self.keyboard_aug_char_p,
                    aug_word_p=self.keyboard_aug_word_p,
                    aug_char_min=1,
                    aug_char_max=2,
                    include_special_char=False,
                    include_numeric=False,
                    include_upper_case=False,
                    min_char=self.min_char
                )
                print("[NoiseGenerator] Keyboard augmenter initialized.")
            except Exception as e:
                print(f"[NoiseGenerator] Keyboard augmenter init failed: {e}")
            
            try:
                self._ocr_aug = nac.OcrAug(
                    aug_char_p=self.ocr_aug_char_p,
                    aug_word_p=0.12,
                    min_char=self.min_char
                )
                print("[NoiseGenerator] OCR augmenter initialized.")
            except Exception as e:
                print(f"[NoiseGenerator] OCR augmenter init failed: {e}")
            
            try:
                self._char_aug = nac.RandomCharAug(
                    action="substitute",
                    aug_char_p=self.char_aug_p,
                    aug_word_p=0.10,
                    min_char=self.min_char
                )
                print("[NoiseGenerator] Character augmenter initialized.")
            except Exception as e:
                print(f"[NoiseGenerator] Character augmenter init failed: {e}")
        
        if SPACY_AVAILABLE:
            try:
                self._spacy_nlp = spacy.load("en_core_web_sm")
                print("[NoiseGenerator] spaCy model loaded (en_core_web_sm).")
            except OSError:
                print("[NoiseGenerator] spaCy model 'en_core_web_sm' not found.")
                print("  Install with: python -m spacy download en_core_web_sm")
    
    # =========================================================================
    # Core Noise Application Methods
    # =========================================================================
    
    def apply_keyboard_noise(self, text: str) -> str:
        """
        Apply QWERTY keyboard-distance typo noise.
        
        Simulates real-world fat-finger typing errors by substituting
        characters with their QWERTY-adjacent neighbors.
        
        Args:
            text: Original customer utterance.
            
        Returns:
            Text with keyboard typo noise applied.
        """
        if self._keyboard_aug is not None:
            try:
                result = self._keyboard_aug.augment(text)
                if isinstance(result, list):
                    result = result[0]
                return self._enforce_budget(text, result)
            except Exception:
                pass
        
        # Fallback: manual QWERTY neighbor substitution
        return self._manual_keyboard_noise(text)
    
    def apply_ocr_noise(self, text: str) -> str:
        """
        Apply OCR-style character confusion noise.
        
        Simulates optical character recognition errors commonly seen
        in scanned document transcripts (e.g., 'l' <-> '1', 'O' <-> '0').
        
        Args:
            text: Original customer utterance.
            
        Returns:
            Text with OCR-style noise applied.
        """
        if self._ocr_aug is not None:
            try:
                result = self._ocr_aug.augment(text)
                if isinstance(result, list):
                    result = result[0]
                return self._enforce_budget(text, result)
            except Exception:
                pass
        
        # Fallback: manual OCR confusion
        return self._manual_ocr_noise(text)
    
    def apply_char_noise(self, text: str) -> str:
        """
        Apply random character-level mutations.
        
        Randomly inserts, deletes, swaps, or substitutes characters
        to simulate general typing imperfections.
        
        Args:
            text: Original customer utterance.
            
        Returns:
            Text with character-level noise applied.
        """
        if self._char_aug is not None:
            try:
                result = self._char_aug.augment(text)
                if isinstance(result, list):
                    result = result[0]
                return self._enforce_budget(text, result)
            except Exception:
                pass
        
        # Fallback: manual character mutations
        return self._manual_char_noise(text)
    
    def apply_linguistic_noise(self, text: str) -> str:
        """
        Apply linguistic-level noise using spaCy NLP analysis.
        
        Strategies:
            - Domain-aware synonym substitution (POS-aware)
            - Contraction expansion / contraction toggling
            - Random case variation
        
        Args:
            text: Original customer utterance.
            
        Returns:
            Text with linguistic noise applied.
        """
        result = text
        
        # Strategy 1: Domain synonym substitution
        if self.rng.random() < 0.6:
            result = self._domain_synonym_swap(result)
        
        # Strategy 2: Contraction toggling
        if self.rng.random() < 0.4:
            result = self._toggle_contraction(result)
        
        # Strategy 3: Case variation
        if self.rng.random() < 0.3:
            result = self._case_variation(result)
        
        return result
    
    def apply_combined_noise(self, text: str) -> str:
        """
        Apply a random combination of noise strategies for maximum diversity.
        
        Randomly selects and stacks 2-3 noise types to create highly
        varied but still interpretable transcript variants.
        
        Args:
            text: Original customer utterance.
            
        Returns:
            Text with combined noise applied.
        """
        strategies = [
            self.apply_keyboard_noise,
            self.apply_linguistic_noise,
            self.apply_char_noise,
        ]
        
        # Apply 1-2 random strategies
        self.rng.shuffle(strategies)
        num_to_apply = self.rng.randint(1, 2)
        
        result = text
        for strategy in strategies[:num_to_apply]:
            result = strategy(result)
        
        return self._enforce_budget(text, result)
    
    # =========================================================================
    # Fallback Manual Noise Methods (when NLPAug is not installed)
    # =========================================================================
    
    def _manual_keyboard_noise(self, text: str) -> str:
        """Manual QWERTY keyboard noise without NLPAug."""
        words = text.split()
        modified = []
        
        for word in words:
            # Skip short words and numbers
            if len(word) < self.min_char or (self.preserve_numbers and word.isdigit()):
                modified.append(word)
                continue
            
            if self.rng.random() < self.keyboard_aug_word_p:
                chars = list(word)
                # Pick a random character position to mutate
                pos = self.rng.randint(0, len(chars) - 1)
                char_lower = chars[pos].lower()
                if char_lower in self.QWERTY_NEIGHBORS:
                    neighbors = self.QWERTY_NEIGHBORS[char_lower]
                    replacement = self.rng.choice(list(neighbors))
                    if chars[pos].isupper():
                        replacement = replacement.upper()
                    chars[pos] = replacement
                modified.append("".join(chars))
            else:
                modified.append(word)
        
        return " ".join(modified)
    
    def _manual_ocr_noise(self, text: str) -> str:
        """Manual OCR confusion noise without NLPAug."""
        ocr_confusions = {
            'l': ['1', 'I', '|'], 'I': ['l', '1', '|'],
            '1': ['l', 'I'], '0': ['O', 'o'], 'O': ['0', 'o'],
            'o': ['0', 'O'], 'S': ['5', '$'], '5': ['S'],
            'B': ['8', '3'], '8': ['B'], 'g': ['9', 'q'],
            'q': ['g', '9'], 'Z': ['2'], '2': ['Z'],
            'rn': ['m'], 'cl': ['d'], 'vv': ['w'],
        }
        
        words = text.split()
        modified = []
        
        for word in words:
            if len(word) < self.min_char or (self.preserve_numbers and word.isdigit()):
                modified.append(word)
                continue
            
            if self.rng.random() < 0.12:
                chars = list(word)
                pos = self.rng.randint(0, len(chars) - 1)
                if chars[pos] in ocr_confusions:
                    chars[pos] = self.rng.choice(ocr_confusions[chars[pos]])
                modified.append("".join(chars))
            else:
                modified.append(word)
        
        return " ".join(modified)
    
    def _manual_char_noise(self, text: str) -> str:
        """Manual character mutation noise without NLPAug."""
        words = text.split()
        modified = []
        
        for word in words:
            if len(word) < self.min_char or (self.preserve_numbers and word.isdigit()):
                modified.append(word)
                continue
            
            if self.rng.random() < 0.10:
                chars = list(word)
                action = self.rng.choice(["insert", "delete", "swap", "substitute"])
                
                if action == "insert" and len(chars) > 0:
                    pos = self.rng.randint(0, len(chars))
                    chars.insert(pos, self.rng.choice(string.ascii_lowercase))
                
                elif action == "delete" and len(chars) > 2:
                    pos = self.rng.randint(0, len(chars) - 1)
                    chars.pop(pos)
                
                elif action == "swap" and len(chars) > 1:
                    pos = self.rng.randint(0, len(chars) - 2)
                    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
                
                elif action == "substitute" and len(chars) > 0:
                    pos = self.rng.randint(0, len(chars) - 1)
                    chars[pos] = self.rng.choice(string.ascii_lowercase)
                
                modified.append("".join(chars))
            else:
                modified.append(word)
        
        return " ".join(modified)
    
    def _domain_synonym_swap(self, text: str) -> str:
        """Replace words with domain-specific synonyms."""
        words = text.split()
        modified = words.copy()
        
        indices = list(range(len(words)))
        self.rng.shuffle(indices)
        swaps = 0
        
        for idx in indices:
            word_clean = words[idx].strip(".,!?;:'\"-").lower()
            if word_clean in self.DOMAIN_SYNONYMS:
                synonyms = self.DOMAIN_SYNONYMS[word_clean]
                synonym = self.rng.choice(synonyms)
                trailing = ""
                if words[idx] and words[idx][-1] in ".,!?;:":
                    trailing = words[idx][-1]
                modified[idx] = synonym + trailing
                swaps += 1
                if swaps >= 2:
                    break
        
        return " ".join(modified)
    
    def _toggle_contraction(self, text: str) -> str:
        """Expand or contract one contraction in the text."""
        text_lower = text.lower()
        
        # Try expanding
        for contraction, expansion in self.CONTRACTIONS.items():
            if contraction in text_lower:
                pattern = re.compile(re.escape(contraction), re.IGNORECASE)
                return pattern.sub(expansion, text, count=1)
        
        # Try contracting
        reverse_map = {v.lower(): k for k, v in self.CONTRACTIONS.items()}
        for expansion, contraction in reverse_map.items():
            if expansion in text_lower:
                pattern = re.compile(re.escape(expansion), re.IGNORECASE)
                return pattern.sub(contraction, text, count=1)
        
        return text
    
    def _case_variation(self, text: str) -> str:
        """Apply random case variations."""
        strategy = self.rng.choice(["lower", "sentence", "mixed"])
        if strategy == "lower":
            return text.lower()
        elif strategy == "sentence":
            return text[0].upper() + text[1:].lower() if text else text
        else:
            # Random mixed case on a few words
            words = text.split()
            for i in range(len(words)):
                if self.rng.random() < 0.2:
                    words[i] = words[i].lower()
            return " ".join(words)
    
    # =========================================================================
    # Budget Enforcement
    # =========================================================================
    
    def _enforce_budget(self, original: str, augmented: str) -> str:
        """
        Enforce the noise budget constraint.
        
        If the augmented text exceeds the maximum allowed perturbation
        rate, it is clipped to stay within budget.
        
        Args:
            original: Original text before augmentation.
            augmented: Text after augmentation.
            
        Returns:
            Augmented text, potentially reverted if budget exceeded.
        """
        # Character-level budget check
        if len(original) > 0:
            char_changes = sum(
                1 for a, b in zip(original, augmented) if a != b
            )
            char_changes += abs(len(original) - len(augmented))
            char_rate = char_changes / len(original)
            
            if char_rate > self.max_char_perturbation_rate:
                return original  # Revert if budget exceeded
        
        # Word-level budget check
        orig_words = original.split()
        aug_words = augmented.split()
        if len(orig_words) > 0:
            word_changes = sum(
                1 for a, b in zip(orig_words, aug_words) if a.lower() != b.lower()
            )
            word_changes += abs(len(orig_words) - len(aug_words))
            word_rate = word_changes / len(orig_words)
            
            if word_rate > self.max_word_perturbation_rate:
                return original  # Revert if budget exceeded
        
        return augmented
    
    def get_config(self) -> Dict[str, Any]:
        """Return current noise generator configuration."""
        return {
            "seed": self.seed,
            "keyboard_aug_available": self._keyboard_aug is not None,
            "ocr_aug_available": self._ocr_aug is not None,
            "char_aug_available": self._char_aug is not None,
            "spacy_available": self._spacy_nlp is not None,
            "nlpaug_installed": NLPAUG_AVAILABLE,
            "spacy_installed": SPACY_AVAILABLE,
            "max_char_perturbation_rate": self.max_char_perturbation_rate,
            "max_word_perturbation_rate": self.max_word_perturbation_rate,
            "preserve_numbers": self.preserve_numbers,
        }


if __name__ == "__main__":
    print("=" * 60)
    print("  IntentAugment AI - Noise Generator Module Demo")
    print("=" * 60)
    
    ng = NoiseGenerator(seed=42)
    
    test_sentences = [
        "I need a refund for my damaged order",
        "Can you cancel my order immediately?",
        "Where is my package? It hasn't arrived yet.",
        "My payment was declined at checkout",
        "I can't login to my account",
    ]
    
    noise_types = [
        ("Keyboard Noise", ng.apply_keyboard_noise),
        ("OCR Noise",      ng.apply_ocr_noise),
        ("Char Noise",     ng.apply_char_noise),
        ("Linguistic",     ng.apply_linguistic_noise),
        ("Combined",       ng.apply_combined_noise),
    ]
    
    for sentence in test_sentences:
        print(f"\nOriginal: {sentence}")
        for name, func in noise_types:
            result = func(sentence)
            changed = "CHANGED" if result != sentence else "SAME"
            print(f"  [{name:15s}] {result}  ({changed})")
    
    print("\n" + "=" * 60)
    print("Configuration:", json.dumps(ng.get_config(), indent=2) if 'json' in dir() else ng.get_config())

# Need json for config printing in __main__
import json
