"""Keras LSTM model definition and training hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters for a word-level next-token LSTM."""

    sequence_length: int = 20
    embedding_dim: int = 128
    lstm_units: int = 256
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 128
    max_epochs: int = 30
    lstm_layers: int = 2
    early_stopping_patience: int = 4


def build_model(vocabulary_size: int, config: ModelConfig) -> tf.keras.Sequential:
    """Build and compile an Embedding-LSTM-Dropout next-word classifier."""
    if vocabulary_size < 3:
        raise ValueError("The vocabulary must include padding, unknown, and word tokens")
    if config.lstm_layers < 1:
        raise ValueError("At least one LSTM layer is required")

    model_layers: list[tf.keras.layers.Layer] = [
        tf.keras.layers.Input(shape=(config.sequence_length,), dtype="int32"),
        tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=config.embedding_dim,
            name="word_embedding",
        ),
    ]
    for layer_index in range(config.lstm_layers):
        model_layers.append(
            tf.keras.layers.LSTM(
                config.lstm_units,
                return_sequences=layer_index < config.lstm_layers - 1,
                name=f"lstm_{layer_index + 1}",
            )
        )
        model_layers.append(
            tf.keras.layers.Dropout(config.dropout_rate, name=f"dropout_{layer_index + 1}")
        )
    model_layers.append(
        tf.keras.layers.Dense(vocabulary_size, activation="softmax", name="next_word")
    )

    model = tf.keras.Sequential(model_layers, name="shakespeare_lstm")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model