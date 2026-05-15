from __future__ import annotations

import argparse
from collections.abc import Iterator

import numpy as np

from captioning.scratch_decoder import (
    DenseWeights,
    RecurrentWeights,
    ScratchDecoder,
)


def build_tiny_decoder(model_kind: str, seed: int) -> ScratchDecoder:
    rng = np.random.default_rng(seed)
    feature_dim = 3
    embed_dim = 4
    hidden_units = 3
    vocab_size = 6
    recurrent_output_dim = 4 * hidden_units if model_kind == "lstm" else hidden_units

    return ScratchDecoder(
        model_kind=model_kind,
        image_projection=DenseWeights(
            kernel=rng.normal(0.0, 0.2, size=(feature_dim, embed_dim)),
            bias=rng.normal(0.0, 0.2, size=(embed_dim,)),
        ),
        token_embedding=rng.normal(0.0, 0.2, size=(vocab_size, embed_dim)),
        recurrent_layers=(
            RecurrentWeights(
                kernel=rng.normal(0.0, 0.2, size=(embed_dim, recurrent_output_dim)),
                recurrent_kernel=rng.normal(0.0, 0.2, size=(hidden_units, recurrent_output_dim)),
                bias=rng.normal(0.0, 0.2, size=(recurrent_output_dim,)),
            ),
        ),
        token_distribution=DenseWeights(
            kernel=rng.normal(0.0, 0.2, size=(hidden_units, vocab_size)),
            bias=rng.normal(0.0, 0.2, size=(vocab_size,)),
        ),
        word_to_idx={"<pad>": 0, "<start>": 1, "<end>": 2, "a": 3, "b": 4, "c": 5},
        idx_to_word={str(index): token for token, index in {"<pad>": 0, "<start>": 1, "<end>": 2, "a": 3, "b": 4, "c": 5}.items()},
        max_steps=3,
    )


def make_batch(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    features = rng.normal(0.0, 0.5, size=(2, 3))
    tokens = np.array(
        [
            [1, 3, 4],
            [1, 5, 0],
        ],
        dtype=np.int64,
    )
    targets = np.array(
        [
            [3, 4, 2],
            [5, 2, 0],
        ],
        dtype=np.int64,
    )
    _ = rng
    return features, tokens, targets, 0


def parameter_checks(decoder: ScratchDecoder) -> Iterator[tuple[str, np.ndarray, np.ndarray, tuple[int, ...]]]:
    _, gradients = decoder.loss_and_gradients_batch(*make_batch(seed=7))
    yield "image_projection.kernel", decoder.image_projection.kernel, gradients.image_projection.kernel, (1, 2)
    yield "image_projection.bias", decoder.image_projection.bias, gradients.image_projection.bias, (2,)
    yield "token_embedding", decoder.token_embedding, gradients.token_embedding, (3, 1)
    yield "recurrent_layers.0.kernel", decoder.recurrent_layers[0].kernel, gradients.recurrent_layers[0].kernel, (2, 1)
    yield "recurrent_layers.0.recurrent_kernel", decoder.recurrent_layers[0].recurrent_kernel, gradients.recurrent_layers[0].recurrent_kernel, (1, 2)
    yield "recurrent_layers.0.bias", decoder.recurrent_layers[0].bias, gradients.recurrent_layers[0].bias, (2,)
    yield "token_distribution.kernel", decoder.token_distribution.kernel, gradients.token_distribution.kernel, (1, 3)
    yield "token_distribution.bias", decoder.token_distribution.bias, gradients.token_distribution.bias, (3,)


def numerical_gradient(
    decoder: ScratchDecoder,
    parameter: np.ndarray,
    index: tuple[int, ...],
    epsilon: float,
) -> float:
    features, tokens, targets, pad_idx = make_batch(seed=7)
    original = float(parameter[index])

    parameter[index] = original + epsilon
    loss_plus, _ = decoder.loss_and_gradients_batch(features, tokens, targets, pad_idx)

    parameter[index] = original - epsilon
    loss_minus, _ = decoder.loss_and_gradients_batch(features, tokens, targets, pad_idx)

    parameter[index] = original
    return (loss_plus - loss_minus) / (2.0 * epsilon)


def relative_error(a: float, b: float) -> float:
    return abs(a - b) / max(1e-12, abs(a) + abs(b))


def run_gradient_check(model_kind: str, epsilon: float, tolerance: float) -> None:
    decoder = build_tiny_decoder(model_kind, seed=123)
    loss, _ = decoder.loss_and_gradients_batch(*make_batch(seed=7))
    print(f"Gradient check: {model_kind}")
    print(f"Initial masked CE loss: {loss:.8f}")

    max_error = 0.0
    for name, parameter, gradient, index in parameter_checks(decoder):
        numerical = numerical_gradient(decoder, parameter, index, epsilon)
        analytical = float(gradient[index])
        error = relative_error(numerical, analytical)
        max_error = max(max_error, error)
        print(
            f"{name}{index}: analytical={analytical:.10f}, "
            f"numerical={numerical:.10f}, rel_error={error:.3e}"
        )

    if max_error > tolerance:
        raise SystemExit(
            f"Gradient check failed: max relative error {max_error:.3e} > {tolerance:.3e}"
        )
    print(f"Gradient check passed: max relative error {max_error:.3e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finite-difference checks for scratch RNN/LSTM backpropagation.")
    parser.add_argument("--model-kind", choices=("rnn", "lstm"), required=True)
    parser.add_argument("--epsilon", type=float, default=1e-5)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_gradient_check(args.model_kind, args.epsilon, args.tolerance)


if __name__ == "__main__":
    main()
