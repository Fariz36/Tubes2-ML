from __future__ import annotations

import argparse

import numpy as np

from captioning.scratch_decoder import (
    SELECTED_EXPERIMENTS,
    load_keras_model,
    load_scratch_decoder,
)
from captioning.scratch_backprop import (
    ScratchGradients,
    load_teacher_forcing_batch,
)


def compare_backprop(experiment_id: str, split: str, batch_size: int) -> None:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow is required for Keras-vs-scratch backprop comparison. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    scratch_decoder = load_scratch_decoder(experiment_id)
    keras_model = load_keras_model(experiment_id)
    feature_batch, tokens, targets, pad_idx = load_teacher_forcing_batch(split, batch_size)

    scratch_loss, scratch_gradients = scratch_decoder.backward_batch(
        feature_batch,
        tokens,
        targets,
        pad_idx,
    )
    keras_loss, keras_gradients = keras_loss_and_gradients(
        tf,
        keras_model,
        feature_batch,
        tokens,
        targets,
        pad_idx,
    )

    scratch_arrays = flatten_scratch_gradients(scratch_gradients)
    print(f"Backprop comparison: {experiment_id} ({split})")
    print(f"Batch size: {len(feature_batch)}")
    print(f"Keras loss:   {keras_loss:.8f}")
    print(f"Scratch loss: {scratch_loss:.8f}")
    print(f"Loss abs diff: {abs(keras_loss - scratch_loss):.3e}")
    print("\nGradient comparison:")

    if len(keras_gradients) != len(scratch_arrays):
        raise ValueError(
            f"Gradient count mismatch: Keras={len(keras_gradients)}, scratch={len(scratch_arrays)}"
        )

    max_relative_error = 0.0
    for (name, scratch_grad), keras_grad in zip(scratch_arrays, keras_gradients, strict=True):
        keras_array = np.asarray(keras_grad, dtype=np.float32)
        if keras_array.shape != scratch_grad.shape:
            raise ValueError(f"Shape mismatch for {name}: Keras={keras_array.shape}, scratch={scratch_grad.shape}")
        max_abs_diff = float(np.max(np.abs(keras_array - scratch_grad)))
        relative = relative_l2_error(keras_array, scratch_grad)
        max_relative_error = max(max_relative_error, relative)
        print(f"{name}: max_abs_diff={max_abs_diff:.3e}, relative_l2_error={relative:.3e}")

    print(f"\nMax relative L2 error: {max_relative_error:.3e}")


def compare_backprop_and_update(
    experiment_id: str,
    split: str,
    batch_size: int,
    learning_rate: float,
) -> None:
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow is required for Keras-vs-scratch weight update comparison. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    scratch_decoder = load_scratch_decoder(experiment_id)
    keras_model = load_keras_model(experiment_id)
    feature_batch, tokens, targets, pad_idx = load_teacher_forcing_batch(split, batch_size)

    scratch_loss, updated_scratch_decoder, scratch_gradients = scratch_decoder.train_batch(
        feature_batch,
        tokens,
        targets,
        pad_idx,
        learning_rate,
    )
    keras_loss, keras_gradients = keras_sgd_update(
        tf,
        keras_model,
        feature_batch,
        tokens,
        targets,
        pad_idx,
        learning_rate,
    )

    scratch_gradient_arrays = flatten_scratch_gradients(scratch_gradients)
    scratch_weight_arrays = flatten_scratch_weights(updated_scratch_decoder)
    keras_weight_arrays = [variable.numpy() for variable in keras_model.trainable_variables]

    print(f"Backprop + one-step SGD comparison: {experiment_id} ({split})")
    print(f"Batch size: {len(feature_batch)}")
    print(f"Learning rate: {learning_rate}")
    print(f"Keras loss:   {keras_loss:.8f}")
    print(f"Scratch loss: {scratch_loss:.8f}")
    print(f"Loss abs diff: {abs(keras_loss - scratch_loss):.3e}")
    print("\nGradient and updated-weight comparison:")

    if len(keras_gradients) != len(scratch_gradient_arrays):
        raise ValueError(
            f"Gradient count mismatch: Keras={len(keras_gradients)}, scratch={len(scratch_gradient_arrays)}"
        )
    if len(keras_weight_arrays) != len(scratch_weight_arrays):
        raise ValueError(
            f"Weight count mismatch: Keras={len(keras_weight_arrays)}, scratch={len(scratch_weight_arrays)}"
        )

    max_gradient_error = 0.0
    max_weight_error = 0.0
    for ((name, scratch_grad), keras_grad, (_, scratch_weight), keras_weight) in zip(
        scratch_gradient_arrays,
        keras_gradients,
        scratch_weight_arrays,
        keras_weight_arrays,
        strict=True,
    ):
        keras_grad = np.asarray(keras_grad, dtype=np.float32)
        keras_weight = np.asarray(keras_weight, dtype=np.float32)
        if keras_grad.shape != scratch_grad.shape:
            raise ValueError(f"Gradient shape mismatch for {name}: Keras={keras_grad.shape}, scratch={scratch_grad.shape}")
        if keras_weight.shape != scratch_weight.shape:
            raise ValueError(f"Weight shape mismatch for {name}: Keras={keras_weight.shape}, scratch={scratch_weight.shape}")

        gradient_error = relative_l2_error(keras_grad, scratch_grad)
        weight_error = relative_l2_error(keras_weight, scratch_weight)
        max_gradient_error = max(max_gradient_error, gradient_error)
        max_weight_error = max(max_weight_error, weight_error)
        print(
            f"{name}: gradient_rel_error={gradient_error:.3e}, "
            f"updated_weight_rel_error={weight_error:.3e}"
        )

    print(f"\nMax gradient relative L2 error: {max_gradient_error:.3e}")
    print(f"Max updated-weight relative L2 error: {max_weight_error:.3e}")


def keras_loss_and_gradients(tf, model, feature_batch, tokens, targets, pad_idx):
    feature_tensor = tf.convert_to_tensor(feature_batch, dtype=tf.float32)
    token_tensor = tf.convert_to_tensor(tokens, dtype=tf.int32)
    target_tensor = tf.convert_to_tensor(targets, dtype=tf.int32)

    with tf.GradientTape() as tape:
        predictions = model([feature_tensor, token_tensor], training=True)
        losses = tf.keras.losses.sparse_categorical_crossentropy(target_tensor, predictions)
        mask = tf.cast(tf.not_equal(target_tensor, pad_idx), losses.dtype)
        loss = tf.math.divide_no_nan(tf.reduce_sum(losses * mask), tf.reduce_sum(mask))

    gradients = tape.gradient(loss, model.trainable_variables)
    dense_gradients = []
    for gradient in gradients:
        if isinstance(gradient, tf.IndexedSlices):
            gradient = tf.convert_to_tensor(gradient)
        dense_gradients.append(gradient.numpy())
    return float(loss.numpy()), dense_gradients


def keras_sgd_update(tf, model, feature_batch, tokens, targets, pad_idx, learning_rate: float):
    loss, gradients = keras_loss_and_gradients(
        tf,
        model,
        feature_batch,
        tokens,
        targets,
        pad_idx,
    )
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables, strict=True))
    return loss, gradients


def flatten_scratch_gradients(gradients: ScratchGradients) -> list[tuple[str, np.ndarray]]:
    arrays = [
        ("image_projection.kernel", gradients.image_projection.kernel),
        ("image_projection.bias", gradients.image_projection.bias),
        ("token_embedding", gradients.token_embedding),
    ]
    for index, grad in enumerate(gradients.recurrent_layers):
        arrays.extend(
            [
                (f"recurrent_layers.{index}.kernel", grad.kernel),
                (f"recurrent_layers.{index}.recurrent_kernel", grad.recurrent_kernel),
                (f"recurrent_layers.{index}.bias", grad.bias),
            ]
        )
    arrays.extend(
        [
            ("token_distribution.kernel", gradients.token_distribution.kernel),
            ("token_distribution.bias", gradients.token_distribution.bias),
        ]
    )
    return arrays


def flatten_scratch_weights(decoder) -> list[tuple[str, np.ndarray]]:
    arrays = [
        ("image_projection.kernel", decoder.image_projection.kernel),
        ("image_projection.bias", decoder.image_projection.bias),
        ("token_embedding", decoder.token_embedding),
    ]
    for index, weights in enumerate(decoder.recurrent_layers):
        arrays.extend(
            [
                (f"recurrent_layers.{index}.kernel", weights.kernel),
                (f"recurrent_layers.{index}.recurrent_kernel", weights.recurrent_kernel),
                (f"recurrent_layers.{index}.bias", weights.bias),
            ]
        )
    arrays.extend(
        [
            ("token_distribution.kernel", decoder.token_distribution.kernel),
            ("token_distribution.bias", decoder.token_distribution.bias),
        ]
    )
    return arrays


def relative_l2_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(1e-12, np.linalg.norm(a) + np.linalg.norm(b)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Keras and NumPy scratch backpropagation gradients.")
    parser.add_argument("--experiment-id", choices=sorted(SELECTED_EXPERIMENTS), default="lstm_layers1_hidden128")
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--compare-weight-update", action="store_true", help="Also compare one Keras SGD update against one scratch SGD update.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="SGD learning rate for --compare-weight-update.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.compare_weight_update:
        compare_backprop_and_update(args.experiment_id, args.split, args.batch_size, args.learning_rate)
    else:
        compare_backprop(args.experiment_id, args.split, args.batch_size)


if __name__ == "__main__":
    main()
