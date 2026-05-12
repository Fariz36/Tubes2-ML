from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_DIR = REPO_ROOT / "artifacts" / "captioning" / "experiments"
FEATURE_DIR = REPO_ROOT / "artifacts" / "captioning" / "features"
MODEL_DIR = REPO_ROOT / "artifacts" / "captioning" / "models"
PREPROCESSED_DIR = REPO_ROOT / "artifacts" / "captioning" / "preprocessed"
SPLITS_DIR = REPO_ROOT / "artifacts" / "captioning" / "splits"
TEACHER_FORCING_DIR = REPO_ROOT / "artifacts" / "captioning" / "teacher_forcing"

FEATURE_BASENAME = "inception_v3_flickr8k"
SELECTED_EXPERIMENTS = {"lstm_layers1_hidden128", "rnn_layers2_hidden128"}


@dataclass(frozen=True)
class DenseWeights:
    kernel: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class RecurrentWeights:
    kernel: np.ndarray
    recurrent_kernel: np.ndarray
    bias: np.ndarray


@dataclass(frozen=True)
class ScratchDecoder:
    model_kind: str
    image_projection: DenseWeights
    token_embedding: np.ndarray
    recurrent_layers: tuple[RecurrentWeights, ...]
    token_distribution: DenseWeights
    word_to_idx: dict[str, int]
    idx_to_word: dict[str, str]
    max_steps: int

    def forward(self, feature: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        projected_image = np.tanh(feature @ self.image_projection.kernel + self.image_projection.bias)
        token_embeddings = self.token_embedding[tokens]
        x = np.concatenate([projected_image[None, :], token_embeddings], axis=0)

        for weights in self.recurrent_layers:
            if self.model_kind == "rnn":
                x = simple_rnn_forward(x, weights)
            elif self.model_kind == "lstm":
                x = lstm_forward(x, weights)
            else:
                raise ValueError(f"Unsupported model_kind: {self.model_kind}")

        caption_steps = x[1:, :]
        logits = caption_steps @ self.token_distribution.kernel + self.token_distribution.bias
        return softmax(logits)

    def greedy_decode(self, feature: np.ndarray) -> str:
        pad_idx = self.word_to_idx["<pad>"]
        start_idx = self.word_to_idx["<start>"]
        end_idx = self.word_to_idx["<end>"]
        tokens = np.full((self.max_steps,), pad_idx, dtype=np.int32)
        tokens[0] = start_idx
        generated: list[str] = []

        for step in range(self.max_steps):
            predictions = self.forward(feature, tokens)
            next_idx = int(np.argmax(predictions[step]))
            if next_idx == end_idx:
                break

            word = self.idx_to_word.get(str(next_idx), "<unk>")
            if word not in {"<pad>", "<start>", "<unk>"}:
                generated.append(word)
            if step + 1 < self.max_steps:
                tokens[step + 1] = next_idx

        return " ".join(generated) if generated else "<empty>"

    def forward_batch(self, feature_batch: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        projected_image = np.tanh(feature_batch @ self.image_projection.kernel + self.image_projection.bias)
        token_embeddings = self.token_embedding[tokens]
        x = np.concatenate([projected_image[:, None, :], token_embeddings], axis=1)

        for weights in self.recurrent_layers:
            if self.model_kind == "rnn":
                x = simple_rnn_forward_batch(x, weights)
            elif self.model_kind == "lstm":
                x = lstm_forward_batch(x, weights)
            else:
                raise ValueError(f"Unsupported model_kind: {self.model_kind}")

        caption_steps = x[:, 1:, :]
        logits = caption_steps @ self.token_distribution.kernel + self.token_distribution.bias
        return softmax(logits)

    def greedy_decode_batch(self, feature_batch: np.ndarray) -> list[str]:
        pad_idx = self.word_to_idx["<pad>"]
        start_idx = self.word_to_idx["<start>"]
        end_idx = self.word_to_idx["<end>"]
        batch_size = feature_batch.shape[0]
        tokens = np.full((batch_size, self.max_steps), pad_idx, dtype=np.int32)
        tokens[:, 0] = start_idx
        generated: list[list[str]] = [[] for _ in range(batch_size)]
        finished = np.zeros((batch_size,), dtype=bool)

        for step in range(self.max_steps):
            predictions = self.forward_batch(feature_batch, tokens)
            next_indices = np.argmax(predictions[:, step, :], axis=1).astype(np.int32)

            for row_index, next_idx in enumerate(next_indices):
                if finished[row_index]:
                    continue
                if next_idx == end_idx:
                    finished[row_index] = True
                    continue

                word = self.idx_to_word.get(str(int(next_idx)), "<unk>")
                if word not in {"<pad>", "<start>", "<unk>"}:
                    generated[row_index].append(word)
                if step + 1 < self.max_steps:
                    tokens[row_index, step + 1] = next_idx

            if finished.all():
                break

        return [" ".join(words) if words else "<empty>" for words in generated]


def simple_rnn_forward(sequence: np.ndarray, weights: RecurrentWeights) -> np.ndarray:
    hidden_units = weights.recurrent_kernel.shape[0]
    hidden = np.zeros((hidden_units,), dtype=np.float32)
    outputs = []
    for timestep in sequence:
        hidden = np.tanh(timestep @ weights.kernel + hidden @ weights.recurrent_kernel + weights.bias)
        outputs.append(hidden)
    return np.stack(outputs, axis=0)


def simple_rnn_forward_batch(sequence: np.ndarray, weights: RecurrentWeights) -> np.ndarray:
    batch_size = sequence.shape[0]
    hidden_units = weights.recurrent_kernel.shape[0]
    hidden = np.zeros((batch_size, hidden_units), dtype=np.float32)
    outputs = []
    for step in range(sequence.shape[1]):
        hidden = np.tanh(sequence[:, step, :] @ weights.kernel + hidden @ weights.recurrent_kernel + weights.bias)
        outputs.append(hidden)
    return np.stack(outputs, axis=1)


def lstm_forward(sequence: np.ndarray, weights: RecurrentWeights) -> np.ndarray:
    hidden_units = weights.recurrent_kernel.shape[0]
    hidden = np.zeros((hidden_units,), dtype=np.float32)
    cell = np.zeros((hidden_units,), dtype=np.float32)
    outputs = []
    for timestep in sequence:
        gates = timestep @ weights.kernel + hidden @ weights.recurrent_kernel + weights.bias
        input_gate, forget_gate, cell_candidate, output_gate = np.split(gates, 4)
        input_gate = sigmoid(input_gate)
        forget_gate = sigmoid(forget_gate)
        cell_candidate = np.tanh(cell_candidate)
        output_gate = sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * cell_candidate
        hidden = output_gate * np.tanh(cell)
        outputs.append(hidden)
    return np.stack(outputs, axis=0)


def lstm_forward_batch(sequence: np.ndarray, weights: RecurrentWeights) -> np.ndarray:
    batch_size = sequence.shape[0]
    hidden_units = weights.recurrent_kernel.shape[0]
    hidden = np.zeros((batch_size, hidden_units), dtype=np.float32)
    cell = np.zeros((batch_size, hidden_units), dtype=np.float32)
    outputs = []
    for step in range(sequence.shape[1]):
        gates = sequence[:, step, :] @ weights.kernel + hidden @ weights.recurrent_kernel + weights.bias
        input_gate, forget_gate, cell_candidate, output_gate = np.split(gates, 4, axis=1)
        input_gate = sigmoid(input_gate)
        forget_gate = sigmoid(forget_gate)
        cell_candidate = np.tanh(cell_candidate)
        output_gate = sigmoid(output_gate)
        cell = forget_gate * cell + input_gate * cell_candidate
        hidden = output_gate * np.tanh(cell)
        outputs.append(hidden)
    return np.stack(outputs, axis=1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_summary_row(experiment_id: str) -> dict[str, object]:
    summary = load_json(EXPERIMENT_DIR / "decoder_training_summary.json")
    for row in summary:
        if row["experiment_id"] == experiment_id:
            return dict(row)
    raise ValueError(f"Unknown experiment_id: {experiment_id}")


def load_scratch_decoder(experiment_id: str) -> ScratchDecoder:
    if experiment_id not in SELECTED_EXPERIMENTS:
        raise ValueError(f"Scratch decoder is currently prepared for: {sorted(SELECTED_EXPERIMENTS)}")

    row = load_summary_row(experiment_id)
    weights_path = MODEL_DIR / f"{experiment_id}.weights.h5"
    word_to_idx = {str(word): int(index) for word, index in load_json(PREPROCESSED_DIR / "word_to_idx.json").items()}
    idx_to_word = {str(index): str(word) for index, word in load_json(PREPROCESSED_DIR / "idx_to_word.json").items()}
    max_steps = int(load_json(TEACHER_FORCING_DIR / "teacher_forcing_metadata.json")["decoder_timesteps"])

    with h5py.File(weights_path, "r") as weights_file:
        recurrent_layers = tuple(
            load_recurrent_layer(weights_file, str(row["model_kind"]), layer_index)
            for layer_index in range(int(row["recurrent_layers"]))
        )
        return ScratchDecoder(
            model_kind=str(row["model_kind"]),
            image_projection=load_dense(weights_file, "layers/dense"),
            token_embedding=np.array(weights_file["layers/embedding/vars/0"], dtype=np.float32),
            recurrent_layers=recurrent_layers,
            token_distribution=load_dense(weights_file, "layers/dense_1"),
            word_to_idx=word_to_idx,
            idx_to_word=idx_to_word,
            max_steps=max_steps,
        )


def load_dense(weights_file: h5py.File, layer_path: str) -> DenseWeights:
    return DenseWeights(
        kernel=np.array(weights_file[f"{layer_path}/vars/0"], dtype=np.float32),
        bias=np.array(weights_file[f"{layer_path}/vars/1"], dtype=np.float32),
    )


def load_recurrent_layer(weights_file: h5py.File, model_kind: str, layer_index: int) -> RecurrentWeights:
    base_name = "simple_rnn" if model_kind == "rnn" else "lstm"
    layer_name = base_name if layer_index == 0 else f"{base_name}_{layer_index}"
    layer_path = f"layers/{layer_name}/cell/vars"
    return RecurrentWeights(
        kernel=np.array(weights_file[f"{layer_path}/0"], dtype=np.float32),
        recurrent_kernel=np.array(weights_file[f"{layer_path}/1"], dtype=np.float32),
        bias=np.array(weights_file[f"{layer_path}/2"], dtype=np.float32),
    )


def load_feature_inputs(split: str) -> tuple[np.ndarray, dict[str, int], dict[str, list[str]]]:
    features = np.load(FEATURE_DIR / f"{FEATURE_BASENAME}_features.npy").astype("float32", copy=False)
    image_ids = load_json(FEATURE_DIR / f"{FEATURE_BASENAME}_image_ids.json")
    image_id_to_index = {str(image_id): index for index, image_id in enumerate(image_ids)}
    captions = load_json(SPLITS_DIR / f"{split}_captions.json")
    return features, image_id_to_index, captions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected captioning decoders with NumPy scratch forward propagation.")
    parser.add_argument("--experiment-id", choices=sorted(SELECTED_EXPERIMENTS), default="lstm_layers1_hidden128")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--compare-keras", action="store_true", help="Print Keras and scratch greedy captions side by side.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decoder = load_scratch_decoder(args.experiment_id)
    features, image_id_to_index, captions = load_feature_inputs(args.split)
    keras_model = load_keras_model(args.experiment_id) if args.compare_keras else None

    mode = "Keras vs scratch greedy decoding" if args.compare_keras else "Scratch greedy decoding"
    print(f"{mode}: {args.experiment_id} ({args.split})")
    for image_id in list(captions)[:args.count]:
        feature = features[image_id_to_index[image_id]]
        scratch_prediction = decoder.greedy_decode(feature)
        print(f"\nImage: {image_id}")
        if keras_model is not None:
            keras_prediction = keras_greedy_decode(keras_model, feature, decoder)
            print(f"Keras:   {keras_prediction}")
            print(f"Scratch: {scratch_prediction}")
            print(f"Match:   {keras_prediction == scratch_prediction}")
        else:
            print(f"Predicted: {scratch_prediction}")
        print("Ground truth:")
        for caption in captions[image_id]:
            print(f"- {caption}")


def load_keras_model(experiment_id: str):
    from captioning.fafo.training_results import build_model_from_summary

    row = load_summary_row(experiment_id)
    model = build_model_from_summary(row)
    model.load_weights(MODEL_DIR / f"{experiment_id}.weights.h5")
    return model


def keras_greedy_decode(model, feature: np.ndarray, decoder: ScratchDecoder) -> str:
    pad_idx = decoder.word_to_idx["<pad>"]
    start_idx = decoder.word_to_idx["<start>"]
    end_idx = decoder.word_to_idx["<end>"]
    tokens = np.full((decoder.max_steps,), pad_idx, dtype=np.int32)
    tokens[0] = start_idx
    generated: list[str] = []

    for step in range(decoder.max_steps):
        predictions = model.predict([feature[None, :], tokens[None, :]], verbose=0)
        next_idx = int(np.argmax(predictions[0, step]))
        if next_idx == end_idx:
            break

        word = decoder.idx_to_word.get(str(next_idx), "<unk>")
        if word not in {"<pad>", "<start>", "<unk>"}:
            generated.append(word)
        if step + 1 < decoder.max_steps:
            tokens[step + 1] = next_idx

    return " ".join(generated) if generated else "<empty>"


if __name__ == "__main__":
    main()
