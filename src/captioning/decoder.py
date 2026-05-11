from __future__ import annotations

from collections.abc import Sequence

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


DEFAULT_FEATURE_DIM = 2048


def masked_sparse_categorical_crossentropy(
    pad_idx: int,
) -> keras.losses.Loss:
    class MaskedSparseCategoricalCrossentropy(keras.losses.Loss):
        def __init__(self) -> None:
            super().__init__(name="masked_sparse_categorical_crossentropy")
            self._loss = keras.losses.SparseCategoricalCrossentropy(
                reduction="none"
            )

        def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
            losses = self._loss(y_true, y_pred)
            mask = tf.cast(tf.not_equal(y_true, pad_idx), losses.dtype)
            losses = losses * mask
            return tf.math.divide_no_nan(
                tf.reduce_sum(losses),
                tf.reduce_sum(mask),
            )

        def get_config(self) -> dict[str, int]:
            return {"pad_idx": pad_idx}

    return MaskedSparseCategoricalCrossentropy()


def compile_caption_decoder(
    model: keras.Model,
    pad_idx: int,
    learning_rate: float = 1e-3,
) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=masked_sparse_categorical_crossentropy(pad_idx),
    )
    return model


def build_simple_rnn_decoder(
    vocab_size: int,
    decoder_timesteps: int,
    pad_idx: int,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    embed_dim: int = 256,
    hidden_units: int | Sequence[int] = 256,
    learning_rate: float = 1e-3,
    compile_model: bool = True,
) -> keras.Model:
    model = _build_preinject_decoder(
        recurrent_layer=layers.SimpleRNN,
        recurrent_name="simple_rnn",
        vocab_size=vocab_size,
        decoder_timesteps=decoder_timesteps,
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        hidden_units=hidden_units,
    )
    if compile_model:
        compile_caption_decoder(model, pad_idx=pad_idx, learning_rate=learning_rate)
    return model


def build_lstm_decoder(
    vocab_size: int,
    decoder_timesteps: int,
    pad_idx: int,
    feature_dim: int = DEFAULT_FEATURE_DIM,
    embed_dim: int = 256,
    hidden_units: int | Sequence[int] = 256,
    learning_rate: float = 1e-3,
    compile_model: bool = True,
) -> keras.Model:
    model = _build_preinject_decoder(
        recurrent_layer=layers.LSTM,
        recurrent_name="lstm",
        vocab_size=vocab_size,
        decoder_timesteps=decoder_timesteps,
        feature_dim=feature_dim,
        embed_dim=embed_dim,
        hidden_units=hidden_units,
    )
    if compile_model:
        compile_caption_decoder(model, pad_idx=pad_idx, learning_rate=learning_rate)
    return model


def _build_preinject_decoder(
    recurrent_layer: type[layers.Layer],
    recurrent_name: str,
    vocab_size: int,
    decoder_timesteps: int,
    feature_dim: int,
    embed_dim: int,
    hidden_units: int | Sequence[int],
) -> keras.Model:
    units_per_layer = _normalize_hidden_units(hidden_units)

    image_features = keras.Input(
        shape=(feature_dim,),
        name="image_features",
    )
    caption_tokens = keras.Input(
        shape=(decoder_timesteps,),
        dtype="int32",
        name="caption_tokens",
    )

    projected_image = layers.Dense(
        embed_dim,
        activation="tanh",
        name="image_projection",
    )(image_features)
    image_timestep = layers.Reshape((1, embed_dim), name="image_timestep")(
        projected_image
    )
    token_embeddings = layers.Embedding(
        vocab_size,
        embed_dim,
        mask_zero=False,
        name="token_embedding",
    )(caption_tokens)
    decoder_inputs = layers.Concatenate(axis=1, name="preinject_sequence")(
        [image_timestep, token_embeddings]
    )

    x = decoder_inputs
    for layer_index, units in enumerate(units_per_layer, start=1):
        x = recurrent_layer(
            units,
            return_sequences=True,
            name=f"{recurrent_name}_{layer_index}",
        )(x)

    caption_steps = layers.Lambda(
        lambda values: values[:, 1:, :],
        name="drop_image_timestep",
    )(x)
    token_distribution = layers.Dense(
        vocab_size,
        activation="softmax",
        name="token_distribution",
    )(caption_steps)

    return keras.Model(
        inputs=[image_features, caption_tokens],
        outputs=token_distribution,
        name=f"preinject_{recurrent_name}_decoder",
    )


def _normalize_hidden_units(hidden_units: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(hidden_units, int):
        units = (hidden_units,)
    else:
        units = tuple(hidden_units)

    if not units:
        raise ValueError("hidden_units must contain at least one layer")
    if any(unit <= 0 for unit in units):
        raise ValueError("hidden_units values must be positive")
    return units
