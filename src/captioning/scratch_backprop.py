from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from captioning.scratch_decoder import (
    FEATURE_BASENAME,
    FEATURE_DIR,
    REPO_ROOT,
    TEACHER_FORCING_DIR,
    DenseWeights,
    RecurrentWeights,
    ScratchDecoder,
)
from captioning.scratch_forward import sigmoid, softmax


SCRATCH_MODEL_DIR = REPO_ROOT / "artifacts" / "captioning" / "scratch_models"


@dataclass(frozen=True)
class RecurrentGradients:
    kernel: np.ndarray
    recurrent_kernel: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class ScratchGradients:
    image_projection: DenseWeights
    token_embedding: np.ndarray
    recurrent_layers: tuple[RecurrentGradients, ...]
    token_distribution: DenseWeights


def evaluate_batch(
    decoder: ScratchDecoder,
    feature_batch: np.ndarray,
    tokens: np.ndarray,
    targets: np.ndarray,
    pad_idx: int,
) -> float:
    probabilities = decoder.forward_batch(feature_batch, tokens)
    return masked_softmax_cross_entropy_loss(probabilities, targets, pad_idx)


def backward_batch(
    decoder: ScratchDecoder,
    feature_batch: np.ndarray,
    tokens: np.ndarray,
    targets: np.ndarray,
    pad_idx: int,
) -> tuple[float, ScratchGradients]:
    projected_image, image_projection_cache = dense_tanh_forward(
        feature_batch,
        decoder.image_projection,
    )
    token_embeddings = decoder.token_embedding[tokens]
    x = np.concatenate([projected_image[:, None, :], token_embeddings], axis=1)

    recurrent_caches = []
    for weights in decoder.recurrent_layers:
        if decoder.model_kind == "rnn":
            x, cache = simple_rnn_forward_batch_with_cache(x, weights)
        elif decoder.model_kind == "lstm":
            x, cache = lstm_forward_batch_with_cache(x, weights)
        else:
            raise ValueError(f"Unsupported model_kind: {decoder.model_kind}")
        recurrent_caches.append(cache)

    caption_steps = x[:, 1:, :]
    logits = caption_steps @ decoder.token_distribution.kernel + decoder.token_distribution.bias
    loss, dlogits = masked_softmax_cross_entropy_backward(logits, targets, pad_idx)

    token_distribution_grad = DenseWeights(
        kernel=np.einsum("bth,btv->hv", caption_steps, dlogits),
        bias=np.sum(dlogits, axis=(0, 1)),
    )
    dcaption_steps = dlogits @ decoder.token_distribution.kernel.T
    dx = np.zeros_like(x)
    dx[:, 1:, :] = dcaption_steps

    recurrent_grads: list[RecurrentGradients] = []
    for weights, cache in reversed(list(zip(decoder.recurrent_layers, recurrent_caches, strict=True))):
        if decoder.model_kind == "rnn":
            dx, grad = simple_rnn_backward_batch(dx, weights, cache)
        else:
            dx, grad = lstm_backward_batch(dx, weights, cache)
        recurrent_grads.append(grad)
    recurrent_grads.reverse()

    dprojected_image = dx[:, 0, :]
    dtoken_embeddings = dx[:, 1:, :]
    _, image_projection_grad = dense_tanh_backward(
        dprojected_image,
        decoder.image_projection,
        image_projection_cache,
    )

    token_embedding_grad = np.zeros_like(decoder.token_embedding)
    np.add.at(token_embedding_grad, tokens, dtoken_embeddings)

    return loss, ScratchGradients(
        image_projection=image_projection_grad,
        token_embedding=token_embedding_grad,
        recurrent_layers=tuple(recurrent_grads),
        token_distribution=token_distribution_grad,
    )


def train_batch(
    decoder: ScratchDecoder,
    feature_batch: np.ndarray,
    tokens: np.ndarray,
    targets: np.ndarray,
    pad_idx: int,
    learning_rate: float,
) -> tuple[float, ScratchDecoder, ScratchGradients]:
    loss, gradients = backward_batch(decoder, feature_batch, tokens, targets, pad_idx)
    return loss, apply_sgd_update(decoder, gradients, learning_rate), gradients


def dense_tanh_forward(x: np.ndarray, weights: DenseWeights) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    linear = x @ weights.kernel + weights.bias
    output = np.tanh(linear)
    return output, {"input": x, "output": output}


def dense_tanh_backward(
    doutput: np.ndarray,
    weights: DenseWeights,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, DenseWeights]:
    x = cache["input"]
    output = cache["output"]
    dlinear = doutput * (1.0 - output * output)
    return dlinear @ weights.kernel.T, DenseWeights(
        kernel=x.T @ dlinear,
        bias=np.sum(dlinear, axis=0),
    )


def masked_softmax_cross_entropy_backward(
    logits: np.ndarray,
    targets: np.ndarray,
    pad_idx: int,
) -> tuple[float, np.ndarray]:
    probabilities = softmax(logits)
    mask = targets != pad_idx
    valid_count = int(np.sum(mask))
    if valid_count == 0:
        return 0.0, np.zeros_like(logits)

    batch_indices = np.arange(targets.shape[0])[:, None]
    time_indices = np.arange(targets.shape[1])[None, :]
    clipped = np.clip(probabilities[batch_indices, time_indices, targets], 1e-12, 1.0)
    loss = -float(np.sum(np.log(clipped) * mask) / valid_count)

    dlogits = probabilities.copy()
    dlogits[batch_indices, time_indices, targets] -= 1.0
    dlogits *= mask[..., None]
    dlogits /= valid_count
    return loss, dlogits


def masked_softmax_cross_entropy_loss(
    probabilities: np.ndarray,
    targets: np.ndarray,
    pad_idx: int,
) -> float:
    mask = targets != pad_idx
    valid_count = int(np.sum(mask))
    if valid_count == 0:
        return 0.0

    batch_indices = np.arange(targets.shape[0])[:, None]
    time_indices = np.arange(targets.shape[1])[None, :]
    clipped = np.clip(probabilities[batch_indices, time_indices, targets], 1e-12, 1.0)
    return -float(np.sum(np.log(clipped) * mask) / valid_count)


def simple_rnn_forward_batch_with_cache(
    sequence: np.ndarray,
    weights: RecurrentWeights,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    batch_size = sequence.shape[0]
    hidden_units = weights.recurrent_kernel.shape[0]
    hidden = np.zeros((batch_size, hidden_units), dtype=sequence.dtype)
    outputs = []
    previous_hiddens = []
    for step in range(sequence.shape[1]):
        previous_hiddens.append(hidden)
        hidden = np.tanh(sequence[:, step, :] @ weights.kernel + hidden @ weights.recurrent_kernel + weights.bias)
        outputs.append(hidden)
    output = np.stack(outputs, axis=1)
    return output, {
        "input": sequence,
        "previous_hiddens": np.stack(previous_hiddens, axis=1),
        "outputs": output,
    }


def simple_rnn_backward_batch(
    doutputs: np.ndarray,
    weights: RecurrentWeights,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, RecurrentGradients]:
    sequence = cache["input"]
    previous_hiddens = cache["previous_hiddens"]
    outputs = cache["outputs"]

    dsequence = np.zeros_like(sequence)
    dkernel = np.zeros_like(weights.kernel)
    drecurrent_kernel = np.zeros_like(weights.recurrent_kernel)
    dbias = np.zeros_like(weights.bias)
    dh_next = np.zeros((sequence.shape[0], weights.recurrent_kernel.shape[0]), dtype=sequence.dtype)

    for step in range(sequence.shape[1] - 1, -1, -1):
        dh = doutputs[:, step, :] + dh_next
        da = dh * (1.0 - outputs[:, step, :] * outputs[:, step, :])
        dkernel += sequence[:, step, :].T @ da
        drecurrent_kernel += previous_hiddens[:, step, :].T @ da
        dbias += np.sum(da, axis=0)
        dsequence[:, step, :] = da @ weights.kernel.T
        dh_next = da @ weights.recurrent_kernel.T

    return dsequence, RecurrentGradients(
        kernel=dkernel,
        recurrent_kernel=drecurrent_kernel,
        bias=dbias,
    )


def lstm_forward_batch_with_cache(
    sequence: np.ndarray,
    weights: RecurrentWeights,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    batch_size = sequence.shape[0]
    hidden_units = weights.recurrent_kernel.shape[0]
    hidden = np.zeros((batch_size, hidden_units), dtype=sequence.dtype)
    cell = np.zeros((batch_size, hidden_units), dtype=sequence.dtype)
    outputs = []
    previous_hiddens = []
    previous_cells = []
    cells = []
    input_gates = []
    forget_gates = []
    cell_candidates = []
    output_gates = []

    for step in range(sequence.shape[1]):
        previous_hiddens.append(hidden)
        previous_cells.append(cell)
        gates = sequence[:, step, :] @ weights.kernel + hidden @ weights.recurrent_kernel + weights.bias
        input_gate, forget_gate, cell_candidate, output_gate = np.split(gates, 4, axis=1)
        input_gate = sigmoid(input_gate)
        forget_gate = sigmoid(forget_gate)
        cell_candidate = np.tanh(cell_candidate)
        output_gate = sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * cell_candidate
        hidden = output_gate * np.tanh(cell)

        input_gates.append(input_gate)
        forget_gates.append(forget_gate)
        cell_candidates.append(cell_candidate)
        output_gates.append(output_gate)
        cells.append(cell)
        outputs.append(hidden)

    output = np.stack(outputs, axis=1)
    return output, {
        "input": sequence,
        "previous_hiddens": np.stack(previous_hiddens, axis=1),
        "previous_cells": np.stack(previous_cells, axis=1),
        "cells": np.stack(cells, axis=1),
        "input_gates": np.stack(input_gates, axis=1),
        "forget_gates": np.stack(forget_gates, axis=1),
        "cell_candidates": np.stack(cell_candidates, axis=1),
        "output_gates": np.stack(output_gates, axis=1),
        "outputs": output,
    }


def lstm_backward_batch(
    doutputs: np.ndarray,
    weights: RecurrentWeights,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, RecurrentGradients]:
    sequence = cache["input"]
    previous_hiddens = cache["previous_hiddens"]
    previous_cells = cache["previous_cells"]
    cells = cache["cells"]
    input_gates = cache["input_gates"]
    forget_gates = cache["forget_gates"]
    cell_candidates = cache["cell_candidates"]
    output_gates = cache["output_gates"]

    dsequence = np.zeros_like(sequence)
    dkernel = np.zeros_like(weights.kernel)
    drecurrent_kernel = np.zeros_like(weights.recurrent_kernel)
    dbias = np.zeros_like(weights.bias)
    hidden_units = weights.recurrent_kernel.shape[0]
    dh_next = np.zeros((sequence.shape[0], hidden_units), dtype=sequence.dtype)
    dc_next = np.zeros((sequence.shape[0], hidden_units), dtype=sequence.dtype)

    for step in range(sequence.shape[1] - 1, -1, -1):
        cell = cells[:, step, :]
        previous_cell = previous_cells[:, step, :]
        input_gate = input_gates[:, step, :]
        forget_gate = forget_gates[:, step, :]
        cell_candidate = cell_candidates[:, step, :]
        output_gate = output_gates[:, step, :]

        dh = doutputs[:, step, :] + dh_next
        tanh_cell = np.tanh(cell)
        doutput_gate = dh * tanh_cell
        dcell = dh * output_gate * (1.0 - tanh_cell * tanh_cell) + dc_next
        dforget_gate = dcell * previous_cell
        dprevious_cell = dcell * forget_gate
        dinput_gate = dcell * cell_candidate
        dcell_candidate = dcell * input_gate

        dinput_gate *= input_gate * (1.0 - input_gate)
        dforget_gate *= forget_gate * (1.0 - forget_gate)
        dcell_candidate *= 1.0 - cell_candidate * cell_candidate
        doutput_gate *= output_gate * (1.0 - output_gate)
        dgates = np.concatenate(
            [dinput_gate, dforget_gate, dcell_candidate, doutput_gate],
            axis=1,
        )

        dkernel += sequence[:, step, :].T @ dgates
        drecurrent_kernel += previous_hiddens[:, step, :].T @ dgates
        dbias += np.sum(dgates, axis=0)
        dsequence[:, step, :] = dgates @ weights.kernel.T
        dh_next = dgates @ weights.recurrent_kernel.T
        dc_next = dprevious_cell

    return dsequence, RecurrentGradients(
        kernel=dkernel,
        recurrent_kernel=drecurrent_kernel,
        bias=dbias,
    )


def apply_sgd_update(
    decoder: ScratchDecoder,
    gradients: ScratchGradients,
    learning_rate: float,
) -> ScratchDecoder:
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than 0")

    return ScratchDecoder(
        model_kind=decoder.model_kind,
        image_projection=sgd_dense(decoder.image_projection, gradients.image_projection, learning_rate),
        token_embedding=decoder.token_embedding - learning_rate * gradients.token_embedding,
        recurrent_layers=tuple(
            sgd_recurrent(weights, grad, learning_rate)
            for weights, grad in zip(decoder.recurrent_layers, gradients.recurrent_layers, strict=True)
        ),
        token_distribution=sgd_dense(decoder.token_distribution, gradients.token_distribution, learning_rate),
        word_to_idx=decoder.word_to_idx,
        idx_to_word=decoder.idx_to_word,
        max_steps=decoder.max_steps,
    )


def sgd_dense(weights: DenseWeights, gradients: DenseWeights, learning_rate: float) -> DenseWeights:
    return DenseWeights(
        kernel=weights.kernel - learning_rate * gradients.kernel,
        bias=weights.bias - learning_rate * gradients.bias,
    )


def sgd_recurrent(
    weights: RecurrentWeights,
    gradients: RecurrentGradients,
    learning_rate: float,
) -> RecurrentWeights:
    return RecurrentWeights(
        kernel=weights.kernel - learning_rate * gradients.kernel,
        recurrent_kernel=weights.recurrent_kernel - learning_rate * gradients.recurrent_kernel,
        bias=weights.bias - learning_rate * gradients.bias,
    )


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        import json

        return json.load(file)


def load_teacher_forcing_batch(
    split: str,
    batch_size: int,
    start: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")
    if start < 0:
        raise ValueError("start must be non-negative")

    metadata = load_json(TEACHER_FORCING_DIR / "teacher_forcing_metadata.json")
    features = np.load(FEATURE_DIR / f"{FEATURE_BASENAME}_features.npy").astype("float32", copy=False)
    teacher = np.load(TEACHER_FORCING_DIR / f"{split}_teacher_forcing.npz")
    end = start + batch_size
    feature_indices = teacher["feature_indices"][start:end]
    if len(feature_indices) == 0:
        raise ValueError(f"No samples available for split={split!r}, start={start}, batch_size={batch_size}")

    return (
        features[feature_indices],
        teacher["caption_inputs"][start:end],
        teacher["caption_targets"][start:end],
        int(metadata["pad_idx"]),
    )


def iter_teacher_forcing_batches(
    split: str,
    batch_size: int,
    limit_samples: int | None = None,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    metadata = load_json(TEACHER_FORCING_DIR / "teacher_forcing_metadata.json")
    features = np.load(FEATURE_DIR / f"{FEATURE_BASENAME}_features.npy").astype("float32", copy=False)
    teacher = np.load(TEACHER_FORCING_DIR / f"{split}_teacher_forcing.npz")
    total = len(teacher["feature_indices"])
    if limit_samples is not None:
        if limit_samples <= 0:
            raise ValueError("limit_samples must be greater than 0")
        total = min(total, limit_samples)

    pad_idx = int(metadata["pad_idx"])
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        feature_indices = teacher["feature_indices"][start:end]
        yield (
            features[feature_indices],
            teacher["caption_inputs"][start:end],
            teacher["caption_targets"][start:end],
            pad_idx,
            start,
        )


def fit_scratch(
    decoder: ScratchDecoder,
    split: str,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    limit_samples: int | None = None,
) -> tuple[ScratchDecoder, list[dict[str, float]]]:
    if epochs <= 0:
        raise ValueError("epochs must be greater than 0")

    current = decoder
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for feature_batch, tokens, targets, pad_idx, _ in iter_teacher_forcing_batches(split, batch_size, limit_samples):
            loss, current, _ = train_batch(
                current,
                feature_batch,
                tokens,
                targets,
                pad_idx,
                learning_rate,
            )
            losses.append(loss)

        epoch_loss = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": float(epoch), "loss": epoch_loss})
        print(f"Epoch {epoch}/{epochs} - scratch_loss: {epoch_loss:.6f}")

    return current, history


def save_scratch_weights(decoder: ScratchDecoder, output_path: str | Path) -> Path:
    path = Path(output_path)
    if path.suffix == ".h5" or path.name.endswith(".weights.h5"):
        raise ValueError("Refusing to save scratch weights to a Keras .h5 file. Use a separate .npz path.")
    if path.suffix != ".npz":
        raise ValueError("Scratch weights must be saved as a .npz file")

    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "model_kind": np.array(decoder.model_kind),
        "image_projection.kernel": decoder.image_projection.kernel,
        "image_projection.bias": decoder.image_projection.bias,
        "token_embedding": decoder.token_embedding,
        "token_distribution.kernel": decoder.token_distribution.kernel,
        "token_distribution.bias": decoder.token_distribution.bias,
        "max_steps": np.array(decoder.max_steps, dtype=np.int32),
    }
    for index, weights in enumerate(decoder.recurrent_layers):
        arrays[f"recurrent_layers.{index}.kernel"] = weights.kernel
        arrays[f"recurrent_layers.{index}.recurrent_kernel"] = weights.recurrent_kernel
        arrays[f"recurrent_layers.{index}.bias"] = weights.bias

    np.savez_compressed(path, **arrays)
    return path


def default_scratch_weights_path(experiment_id: str) -> Path:
    return SCRATCH_MODEL_DIR / f"{experiment_id}_scratch_sgd.npz"
