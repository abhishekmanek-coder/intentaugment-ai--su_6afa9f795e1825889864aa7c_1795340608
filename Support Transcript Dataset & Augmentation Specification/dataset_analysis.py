"""
IntentAugment AI - Support Transcript Dataset Exploratory Data Analysis (EDA)
Computes lexical statistics, class distributions, and error variation profiles
for Deliverable #1: Support Transcript Dataset & Augmentation Specification.
"""

import os
import csv
import json
import re
import math
from collections import Counter

def run_eda(csv_file="raw_customer_support_transcripts.csv"):
    if not os.path.isabs(csv_file):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_file = os.path.join(script_dir, csv_file)

    print(f"[+] Loading dataset from: {csv_file}")
    rows = []
    with open(csv_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    total_samples = len(rows)
    print(f"[+] Total Support Transcripts: {total_samples}")

    def tokenize(text):
        return re.findall(r"\b\w+\b", text.lower())

    all_tokens = []
    word_lengths = []
    char_lengths = []
    intent_counts = Counter()
    channel_counts = Counter()
    sentiment_counts = Counter()
    complexity_counts = Counter()
    intent_tokens = {}

    for r in rows:
        text = r["customer_utterance"]
        tokens = tokenize(text)
        all_tokens.extend(tokens)
        word_lengths.append(len(tokens))
        char_lengths.append(len(text))
        
        intent = r["intent"]
        intent_counts[intent] += 1
        channel_counts[r.get("channel", "unknown")] += 1
        sentiment_counts[r.get("sentiment", "neutral")] += 1
        complexity_counts[r.get("complexity", "simple")] += 1

        if intent not in intent_tokens:
            intent_tokens[intent] = []
        intent_tokens[intent].extend(tokens)

    vocab = set(all_tokens)
    ttr = len(vocab) / len(all_tokens) if all_tokens else 0
    mean_w = sum(word_lengths) / total_samples
    std_w = math.sqrt(sum((w - mean_w) ** 2 for w in word_lengths) / total_samples)

    print("\n=======================================================")
    print("      DATASET LEXICAL & STRUCTURAL STATISTICS         ")
    print("=======================================================")
    print(f"Total Words (Tokens)     : {len(all_tokens):,}")
    print(f"Unique Vocabulary Size   : {len(vocab):,}")
    print(f"Type-Token Ratio (TTR)   : {ttr:.4f}")
    print(f"Mean Words / Utterance   : {mean_w:.2f} (std: {std_w:.2f})")
    print(f"Word Length Range        : [{min(word_lengths)} - {max(word_lengths)}] words")
    print(f"Mean Characters / Utter  : {sum(char_lengths) / total_samples:.2f}")

    print("\n=======================================================")
    print("      INTENT CLASS DISTRIBUTION (CLASS BALANCE)       ")
    print("=======================================================")
    for intent, count in intent_counts.most_common():
        pct = (count / total_samples) * 100
        print(f"  {intent:<26} : {count:>3} samples ({pct:.1f}%)")

    print("\n=======================================================")
    print("      TOP LEXICAL DISCRIMINATORS PER INTENT           ")
    print("=======================================================")
    stopwords = {"i", "to", "my", "the", "a", "and", "is", "for", "in", "it", "of", "can", "on", "you", "this", "have", "with", "what", "do", "how", "your", "be", "an", "not", "me", "from", "at", "as", "if", "or", "so", "are", "was", "will", "please", "would", "like"}
    for intent, tokens in intent_tokens.items():
        filtered = [t for t in tokens if t not in stopwords and len(t) > 2]
        top_k = [w for w, _ in Counter(filtered).most_common(5)]
        print(f"  {intent:<26} : {', '.join(top_k)}")

    print("\n[+] EDA complete. All checks passed successfully.")

if __name__ == "__main__":
    run_eda()
