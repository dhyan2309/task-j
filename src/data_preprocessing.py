"""Download, clean, tokenize, and window the Shakespeare corpus."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf


GUTENBERG_PAGE = "https://www.gutenberg.org/ebooks/100"
RAW_TEXT_URL = "https://www.gutenberg.org/cache/epub/100/pg100.txt"
PAD_TOKEN = "<PAD>"
UNKNOWN_TOKEN = "<UNK>"


class WordTokenizer:
    """A compact word vocabulary with reserved padding and unknown IDs."""

    def __init__(self, vocabulary: list[str]) -> None:
        self.id_to_word = [PAD_TOKEN, UNKNOWN_TOKEN, *vocabulary]
        self.word_to_id = {word: index for index, word in enumerate(self.id_to_word)}

    @classmethod
    def fit(cls, tokens: list[str], max_vocabulary_size: int = 20_000) -> "WordTokenizer":
        """Build a frequency-ranked vocabulary from training tokens only."""
        if max_vocabulary_size < 3:
            raise ValueError("max_vocabulary_size must be at least 3")
        most_common = Counter(tokens).most_common(max_vocabulary_size - 2)
        vocabulary = [word for word, _ in most_common if word not in {PAD_TOKEN, UNKNOWN_TOKEN}]
        return cls(vocabulary)

    @property
    def vocabulary_size(self) -> int:
        return len(self.id_to_word)

    def encode(self, tokens: list[str]) -> list[int]:
        """Map words to IDs, using the reserved unknown ID for unseen words."""
        unknown_id = self.word_to_id[UNKNOWN_TOKEN]
        return [self.word_to_id.get(token, unknown_id) for token in tokens]

    def decode(self, token_ids: list[int]) -> list[str]:
        """Map IDs to words while omitting padding and unknown markers."""
        return [
            self.id_to_word[token_id]
            for token_id in token_ids
            if 0 < token_id < self.vocabulary_size and token_id != 1
        ]

    def save(self, path: Path) -> None:
        """Persist the ordered vocabulary used by a saved model."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.id_to_word[2:], ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "WordTokenizer":
        """Restore a vocabulary saved with :meth:`save`."""
        vocabulary = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(vocabulary, list) or not all(isinstance(word, str) for word in vocabulary):
            raise ValueError(f"Invalid vocabulary file: {path}")
        return cls(vocabulary)


@dataclass
class PreparedDatasets:
    """Tokenization and chronological train/validation datasets."""

    tokenizer: WordTokenizer
    train_dataset: tf.data.Dataset
    validation_dataset: tf.data.Dataset
    train_token_count: int
    validation_token_count: int


def download_shakespeare(path: Path) -> Path:
    """Download the Project Gutenberg plain-text edition if it is not cached."""
    if path.exists() and path.stat().st_size > 0:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(RAW_TEXT_URL, timeout=(10, 90))
        response.raise_for_status()
        text = response.content.decode("utf-8", errors="replace")
    except requests.RequestException as error:
        raise RuntimeError(
            f"Could not download Shakespeare's Complete Works from {GUTENBERG_PAGE}: {error}"
        ) from error

    if not text.strip():
        raise RuntimeError("Project Gutenberg returned an empty text file")
    path.write_text(text, encoding="utf-8")
    return path


def load_and_clean_text(path: Path) -> str:
    """Load text, remove Gutenberg boilerplate, punctuation, and excess whitespace."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset does not exist: {path}")
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    start_marker = re.search(
        r"\*\*\*\s*START OF (?:THE )?PROJECT GUTENBERG[^\n]*",
        raw_text,
        flags=re.IGNORECASE,
    )
    if start_marker:
        raw_text = raw_text[start_marker.end() :]
    end_marker = re.search(
        r"\*\*\*\s*END OF (?:THE )?PROJECT GUTENBERG",
        raw_text,
        flags=re.IGNORECASE,
    )
    if end_marker:
        raw_text = raw_text[: end_marker.start()]

    # Keep English letters and whitespace; replacing punctuation with spaces avoids joining words.
    cleaned_text = re.sub(r"[^a-z\s]", " ", raw_text.lower())
    return re.sub(r"\s+", " ", cleaned_text).strip()


def make_next_word_dataset(
    token_ids: list[int], sequence_length: int, batch_size: int, shuffle: bool
) -> tf.data.Dataset:
    """Create sliding input windows and the next-word target for each window."""
    if sequence_length < 1 or batch_size < 1:
        raise ValueError("sequence_length and batch_size must be positive")
    if len(token_ids) <= sequence_length:
        raise ValueError("Not enough tokens to create a next-word training example")

    ids = np.asarray(token_ids, dtype=np.int32)
    # Each target is the token immediately after its fixed-length input window.
    return tf.keras.utils.timeseries_dataset_from_array(
        data=ids[:-1],
        targets=ids[sequence_length:],
        sequence_length=sequence_length,
        sequence_stride=1,
        shuffle=shuffle,
        batch_size=batch_size,
    ).prefetch(tf.data.AUTOTUNE)


def prepare_datasets(
    tokens: list[str],
    sequence_length: int,
    batch_size: int,
    max_vocabulary_size: int = 20_000,
    train_fraction: float = 0.9,
) -> PreparedDatasets:
    """Split sequential tokens chronologically and make next-word datasets."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_index = int(len(tokens) * train_fraction)
    train_tokens = tokens[:split_index]
    validation_tokens = tokens[split_index:]
    if not train_tokens or not validation_tokens:
        raise ValueError("The corpus is too small for the requested train/validation split")

    tokenizer = WordTokenizer.fit(train_tokens, max_vocabulary_size)
    train_ids = tokenizer.encode(train_tokens)
    validation_ids = tokenizer.encode(validation_tokens)
    return PreparedDatasets(
        tokenizer=tokenizer,
        train_dataset=make_next_word_dataset(
            train_ids, sequence_length, batch_size, shuffle=True
        ),
        validation_dataset=make_next_word_dataset(
            validation_ids, sequence_length, batch_size, shuffle=False
        ),
        train_token_count=len(train_ids),
        validation_token_count=len(validation_ids),
    )