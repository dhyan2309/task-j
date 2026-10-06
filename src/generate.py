"""Generate word-level Shakespeare-style text from a trained LSTM."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.data_preprocessing import WordTokenizer


def generate_text(
    model: tf.keras.Model,
    tokenizer: WordTokenizer,
    seed_text: str,
    num_words: int,
    temperature: float,
) -> str:
    """Sample ``num_words`` continuations from a normalized seed sentence."""
    if num_words < 0:
        raise ValueError("num_words cannot be negative")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    normalized_seed = re.sub(r"[^a-z\s]", " ", seed_text.lower())
    seed_tokens = normalized_seed.split()
    if not seed_tokens:
        raise ValueError("seed_text must contain at least one word")
    context_length = int(model.input_shape[1])
    if context_length < 1:
        raise ValueError("The model must have a fixed positive input sequence length")

    context = tokenizer.encode(seed_tokens)[-context_length:]
    generated_tokens = seed_tokens[:]
    for _ in range(num_words):
        padding_count = max(0, context_length - len(context))
        padded_context = [context[0]] * padding_count + context
        probabilities = model(
            np.asarray([padded_context], dtype=np.int32), training=False
        ).numpy()[0]

        # Low temperature sharpens the distribution; high temperature makes sampling more random.
        scaled_log_probabilities = np.log(np.clip(probabilities, 1e-12, 1.0)) / temperature
        scaled_log_probabilities[tokenizer.word_to_id["<PAD>"]] = -np.inf
        scaled_log_probabilities[tokenizer.word_to_id["<UNK>"]] = -np.inf
        scaled_log_probabilities -= np.max(scaled_log_probabilities)
        sampling_probabilities = np.exp(scaled_log_probabilities)
        sampling_probabilities /= sampling_probabilities.sum()
        next_token_id = int(np.random.choice(tokenizer.vocabulary_size, p=sampling_probabilities))

        context = (context + [next_token_id])[-context_length:]
        generated_tokens.extend(tokenizer.decode([next_token_id]))

    return " ".join(generated_tokens)


def save_seed_samples(
    model: tf.keras.Model,
    tokenizer: WordTokenizer,
    seeds: list[str],
    output_path: Path,
    num_words: int = 50,
    temperature: float = 0.8,
) -> list[str]:
    """Generate and save one continuation for every supplied seed phrase."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        f"Seed: {seed}\n{generate_text(model, tokenizer, seed, num_words, temperature)}"
        for seed in seeds
    ]
    output_path.write_text("\n\n".join(samples) + "\n", encoding="utf-8")
    return samples