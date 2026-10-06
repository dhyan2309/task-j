"""Train an LSTM, save its best checkpoint, and plot learning curves."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from src.model import ModelConfig, build_model


@dataclass
class TrainingResult:
    """Model, fit history, and metrics from its best validation-loss epoch."""

    model: tf.keras.Model
    history: tf.keras.callbacks.History
    metrics: dict[str, float | int]
    training_seconds: float
    parameter_count: int


def train_model(
    train_dataset: tf.data.Dataset,
    validation_dataset: tf.data.Dataset,
    vocabulary_size: int,
    config: ModelConfig,
    checkpoint_path: Path,
) -> TrainingResult:
    """Fit the model with early stopping and best-validation checkpointing."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    model = build_model(vocabulary_size, config)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
    ]

    start_time = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config.max_epochs,
        callbacks=callbacks,
        shuffle=False,
        verbose=1,
    )
    training_seconds = time.perf_counter() - start_time
    if not checkpoint_path.exists():
        raise RuntimeError(f"Training did not save a model checkpoint to {checkpoint_path}")

    best_epoch_index = min(
        range(len(history.history["val_loss"])),
        key=history.history["val_loss"].__getitem__,
    )
    metrics: dict[str, float | int] = {
        "best_epoch": best_epoch_index + 1,
        "epochs_trained": len(history.history["loss"]),
        "training_loss": float(history.history["loss"][best_epoch_index]),
        "validation_loss": float(history.history["val_loss"][best_epoch_index]),
        "training_accuracy": float(history.history["accuracy"][best_epoch_index]),
        "validation_accuracy": float(history.history["val_accuracy"][best_epoch_index]),
    }
    best_model = tf.keras.models.load_model(checkpoint_path)
    return TrainingResult(
        model=best_model,
        history=history,
        metrics=metrics,
        training_seconds=training_seconds,
        parameter_count=best_model.count_params(),
    )


def print_training_summary(result: TrainingResult) -> None:
    """Print requested loss, accuracy, epoch, and elapsed-time metrics."""
    metrics = result.metrics
    print("\nTraining summary (best validation-loss epoch)")
    print(f"  Training loss:       {metrics['training_loss']:.4f}")
    print(f"  Validation loss:     {metrics['validation_loss']:.4f}")
    print(f"  Training accuracy:   {metrics['training_accuracy']:.4f}")
    print(f"  Validation accuracy: {metrics['validation_accuracy']:.4f}")
    print(f"  Best epoch:          {metrics['best_epoch']}")
    print(f"  Epochs trained:      {metrics['epochs_trained']}")
    print(f"  Training time:       {result.training_seconds:.1f} seconds")
    print(f"  Trainable parameters:{result.parameter_count:>12,}")


def save_training_plots(history: tf.keras.callbacks.History, output_dir: Path) -> None:
    """Save training/validation loss and accuracy curves as PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.history["loss"]) + 1)

    for metric, title, ylabel, filename in (
        ("loss", "Training and Validation Loss", "Sparse categorical crossentropy", "loss_curve.png"),
        ("accuracy", "Training and Validation Accuracy", "Accuracy", "accuracy_curve.png"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, history.history[metric], label="Training")
        axis.plot(epochs, history.history[f"val_{metric}"], label="Validation")
        axis.set(title=title, xlabel="Epoch", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)