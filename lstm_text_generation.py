"""Train and run the Shakespeare LSTM text-generation project."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.data_preprocessing import (
    download_shakespeare,
    load_and_clean_text,
    prepare_datasets,
)
from src.generate import generate_text, save_seed_samples
from src.model import ModelConfig
from src.train import print_training_summary, save_training_plots, train_model


DEFAULT_SEEDS = [
    "to be or not to be",
    "love is",
    "the king",
    "shall i compare thee",
]
DATA_PATH = Path("data/shakespeare.txt")
MODEL_PATH = Path("models/best_lstm_model.keras")
TOKENIZER_PATH = Path("models/tokenizer.json")
OUTPUT_DIR = Path("outputs")


def parse_args() -> argparse.Namespace:
    """Parse supported training and experiment options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--lstm-units", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-vocabulary-size", type=int, default=20_000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--run-experiment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def run_bonus_experiment(
    tokens: list[str],
    base_config: ModelConfig,
    max_vocabulary_size: int,
    random_seed: int,
) -> Path:
    """Train 20-token/one-layer and 50-token/two-layer models for comparison."""
    experiment_rows: list[dict[str, float | int | str]] = []
    experiment_specs = [
        ("Model A", 20, 1),
        ("Model B", 50, 2),
    ]

    for model_name, sequence_length, lstm_layers in experiment_specs:
        print(f"\nRunning {model_name}: sequence length={sequence_length}, LSTM layers={lstm_layers}")
        seed_everything(random_seed)
        prepared = prepare_datasets(
            tokens,
            sequence_length=sequence_length,
            batch_size=base_config.batch_size,
            max_vocabulary_size=max_vocabulary_size,
        )
        config = ModelConfig(
            sequence_length=sequence_length,
            embedding_dim=base_config.embedding_dim,
            lstm_units=base_config.lstm_units,
            dropout_rate=base_config.dropout_rate,
            learning_rate=base_config.learning_rate,
            batch_size=base_config.batch_size,
            max_epochs=base_config.max_epochs,
            lstm_layers=lstm_layers,
            early_stopping_patience=base_config.early_stopping_patience,
        )
        checkpoint = Path("models") / f"{model_name.lower().replace(' ', '_')}_best.keras"
        result = train_model(
            prepared.train_dataset,
            prepared.validation_dataset,
            prepared.tokenizer.vocabulary_size,
            config,
            checkpoint,
        )
        sample = generate_text(
            result.model, prepared.tokenizer, "to be or not to be", 50, 0.8
        )
        experiment_rows.append(
            {
                "model": model_name,
                "sequence_length": sequence_length,
                "lstm_layers": lstm_layers,
                "training_seconds": round(result.training_seconds, 2),
                "best_epoch": result.metrics["best_epoch"],
                "training_loss": result.metrics["training_loss"],
                "validation_loss": result.metrics["validation_loss"],
                "training_accuracy": result.metrics["training_accuracy"],
                "validation_accuracy": result.metrics["validation_accuracy"],
                "parameter_count": result.parameter_count,
                "context_retention": f"Up to {sequence_length} preceding words",
                "generated_text_quality": "Review generated_text_sample; no automatic quality score",
                "generated_text_sample": sample,
            }
        )

    comparison_path = OUTPUT_DIR / "model_comparison.csv"
    comparison_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(experiment_rows).to_csv(comparison_path, index=False)
    print(f"\nSaved bonus experiment results to {comparison_path}")
    return comparison_path


def main() -> None:
    """Download, preprocess, train, evaluate, visualize, and generate samples."""
    args = parse_args()
    if args.epochs < 1 or args.sequence_length < 1 or args.temperature <= 0:
        raise SystemExit("epochs and sequence length must be positive; temperature must be > 0")
    seed_everything(args.seed)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset_path = download_shakespeare(DATA_PATH)
    cleaned_text = load_and_clean_text(dataset_path)
    tokens = cleaned_text.split()
    if len(tokens) < 100:
        raise RuntimeError("The cleaned Shakespeare corpus is unexpectedly small")
    print(f"Loaded {len(tokens):,} cleaned word tokens from {dataset_path}")

    config = ModelConfig(
        sequence_length=args.sequence_length,
        embedding_dim=args.embedding_dim,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        lstm_layers=2,
    )
    prepared = prepare_datasets(
        tokens,
        sequence_length=config.sequence_length,
        batch_size=config.batch_size,
        max_vocabulary_size=args.max_vocabulary_size,
        train_fraction=0.9,
    )
    prepared.tokenizer.save(TOKENIZER_PATH)
    print(
        f"Vocabulary: {prepared.tokenizer.vocabulary_size:,} tokens | "
        f"Chronological split: {prepared.train_token_count:,} train / "
        f"{prepared.validation_token_count:,} validation tokens"
    )

    result = train_model(
        prepared.train_dataset,
        prepared.validation_dataset,
        prepared.tokenizer.vocabulary_size,
        config,
        MODEL_PATH,
    )
    print_training_summary(result)
    save_training_plots(result.history, OUTPUT_DIR)
    print(f"Saved best model to {MODEL_PATH}")
    print(f"Saved vocabulary to {TOKENIZER_PATH}")
    print(f"Saved learning curves under {OUTPUT_DIR}/")

    sample_path = OUTPUT_DIR / "generated_samples.txt"
    save_seed_samples(
        result.model,
        prepared.tokenizer,
        DEFAULT_SEEDS,
        sample_path,
        num_words=50,
        temperature=args.temperature,
    )
    print(f"Generated 50 words for each of {len(DEFAULT_SEEDS)} seeds: {sample_path}")

    if args.run_experiment:
        run_bonus_experiment(tokens, config, args.max_vocabulary_size, args.seed)


if __name__ == "__main__":
    main()