"""
IntentAugment AI - Dataset Loader Module
=========================================
Handles ingestion, validation, and preprocessing of raw customer support
transcript datasets from CSV/JSON sources.

Author: Abhishek Manek
Deliverable: #2 (Week 2) - Transcript Paraphrasing & Noise Augmentation Pipeline
"""

import os
import csv
import json
import re
from typing import List, Dict, Optional, Tuple


class DatasetLoader:
    """
    Loads and preprocesses customer support transcript datasets.
    
    Supports CSV and JSON input formats with configurable column mapping.
    Performs basic validation: missing fields, empty utterances, and
    label consistency checks.
    """
    
    SUPPORTED_FORMATS = {".csv", ".json", ".jsonl"}
    
    def __init__(
        self,
        text_column: str = "customer_utterance",
        label_column: str = "intent",
        id_column: str = "transcript_id",
        encoding: str = "utf-8"
    ):
        """
        Initialize the DatasetLoader.
        
        Args:
            text_column: Name of the column containing customer utterances.
            label_column: Name of the column containing intent labels.
            id_column: Name of the column containing unique transcript IDs.
            encoding: File encoding (default: utf-8).
        """
        self.text_column = text_column
        self.label_column = label_column
        self.id_column = id_column
        self.encoding = encoding
    
    def load(self, file_path: str) -> List[Dict[str, str]]:
        """
        Load transcripts from a file.
        
        Args:
            file_path: Path to the input file (CSV, JSON, or JSONL).
            
        Returns:
            List of transcript dictionaries with standardized keys.
            
        Raises:
            FileNotFoundError: If the input file does not exist.
            ValueError: If the file format is unsupported or data is invalid.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                f"Supported: {', '.join(self.SUPPORTED_FORMATS)}"
            )
        
        if ext == ".csv":
            records = self._load_csv(file_path)
        elif ext == ".json":
            records = self._load_json(file_path)
        elif ext == ".jsonl":
            records = self._load_jsonl(file_path)
        
        # Validate and standardize
        validated = self._validate_records(records)
        print(f"[DatasetLoader] Loaded {len(validated)} valid transcripts from {file_path}")
        return validated
    
    def _load_csv(self, file_path: str) -> List[Dict[str, str]]:
        """Load records from a CSV file."""
        records = []
        with open(file_path, mode="r", encoding=self.encoding, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records
    
    def _load_json(self, file_path: str) -> List[Dict[str, str]]:
        """Load records from a JSON file (expects a list of objects)."""
        with open(file_path, mode="r", encoding=self.encoding) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON file must contain a list of transcript objects.")
        return data
    
    def _load_jsonl(self, file_path: str) -> List[Dict[str, str]]:
        """Load records from a JSON Lines file."""
        records = []
        with open(file_path, mode="r", encoding=self.encoding) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"[DatasetLoader] Warning: Skipping invalid JSON on line {line_num}: {e}")
        return records
    
    def _validate_records(self, records: List[Dict]) -> List[Dict[str, str]]:
        """
        Validate and standardize transcript records.
        
        Checks:
            - Required columns exist (text, label).
            - Text is not empty or whitespace-only.
            - Labels are non-empty strings.
        
        Returns:
            List of validated and standardized records.
        """
        validated = []
        skipped = 0
        
        for i, record in enumerate(records):
            # Check required fields
            text = record.get(self.text_column, "").strip()
            label = record.get(self.label_column, "").strip()
            transcript_id = record.get(self.id_column, f"AUTO_{i+1:04d}").strip()
            
            if not text:
                skipped += 1
                continue
            if not label:
                skipped += 1
                continue
            
            standardized = {
                "transcript_id": transcript_id,
                "text": text,
                "intent": label
            }
            
            # Carry over optional metadata columns
            for key in ["channel", "sentiment", "complexity"]:
                if key in record:
                    standardized[key] = record[key].strip()
            
            validated.append(standardized)
        
        if skipped > 0:
            print(f"[DatasetLoader] Warning: Skipped {skipped} invalid records.")
        
        return validated
    
    def get_label_distribution(self, records: List[Dict[str, str]]) -> Dict[str, int]:
        """
        Compute intent label distribution.
        
        Args:
            records: List of validated transcript records.
            
        Returns:
            Dictionary mapping intent labels to their counts.
        """
        distribution = {}
        for record in records:
            label = record["intent"]
            distribution[label] = distribution.get(label, 0) + 1
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply minimal text preprocessing.
        
        Normalizes whitespace and strips leading/trailing spaces.
        Intentionally preserves original casing, punctuation, and typos
        since those are features the augmentation pipeline operates on.
        
        Args:
            text: Raw customer utterance text.
            
        Returns:
            Lightly preprocessed text string.
        """
        # Normalize multiple spaces to single space
        text = re.sub(r"\s+", " ", text).strip()
        return text


if __name__ == "__main__":
    # Quick test with the Deliverable 1 dataset
    loader = DatasetLoader()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(
        script_dir,
        "..",
        "..",
        "Support Transcript Dataset & Augmentation Specification",
        "raw_customer_support_transcripts.csv"
    )
    
    if os.path.isfile(dataset_path):
        records = loader.load(dataset_path)
        dist = loader.get_label_distribution(records)
        print(f"\nLoaded {len(records)} transcripts")
        print("Intent Distribution:")
        for intent, count in dist.items():
            print(f"  {intent:<28}: {count}")
    else:
        print(f"Dataset not found at: {dataset_path}")
        print("Run from the project root directory.")
