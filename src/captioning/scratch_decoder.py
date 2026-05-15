from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import h5py
except ModuleNotFoundError:
    h5py = None

REPO_ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT_DIR = REPO_ROOT / "artifacts" / "captioning" / "experiments"
FEATURE_DIR = REPO_ROOT / "artifacts" / "captioning" / "features"
MODEL_DIR = REPO_ROOT / "artifacts" / "captioning" / "models"
PREPROCESSED_DIR = REPO_ROOT / "artifacts" / "captioning" / "preprocessed"
SPLITS_DIR = REPO_ROOT / "artifacts" / "captioning" / "splits"
TEACHER_FORCING_DIR = REPO_ROOT / "artifacts" / "captioning" / "teacher_forcing"

FEATURE_BASENAME = "inception_v3_flickr8k"
PREINJECT_EXPERIMENTS = tuple(
    f"{model_kind}_layers{recurrent_layers}_hidden{hidden_units}"
    for model_kind in ("rnn", "lstm")
    for recurrent_layers in (1, 2, 3)
    for hidden_units in (128, 512)
)
SELECTED_EXPERIMENTS = set(PREINJECT_EXPERIMENTS)
PREDICT_MODE = "predict"
SCRATCH_MODES = (PREDICT_MODE, "evaluate", "backward", "train-batch", "fit")


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
class BeamCandidate:
    tokens: np.ndarray
    words: tuple[str, ...]
    log_probability: float
    ended: bool


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
        from captioning.scratch_forward import decoder_forward

        return decoder_forward(self, feature, tokens)

    def greedy_decode(self, feature: np.ndarray) -> str:
        from captioning.scratch_forward import greedy_decode

        return greedy_decode(self, feature)

    def beam_search_decode(self, feature: np.ndarray, beam_width: int = 3) -> str:
        from captioning.scratch_forward import beam_search_decode

        return beam_search_decode(self, feature, beam_width)

    def forward_batch(self, feature_batch: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        from captioning.scratch_forward import decoder_forward_batch

        return decoder_forward_batch(self, feature_batch, tokens)

    def greedy_decode_batch(self, feature_batch: np.ndarray) -> list[str]:
        from captioning.scratch_forward import greedy_decode_batch

        return greedy_decode_batch(self, feature_batch)

    def loss_and_gradients_batch(
        self,
        feature_batch: np.ndarray,
        tokens: np.ndarray,
        targets: np.ndarray,
        pad_idx: int,
    ):
        from captioning.scratch_backprop import backward_batch

        return backward_batch(self, feature_batch, tokens, targets, pad_idx)

    def evaluate_batch(
        self,
        feature_batch: np.ndarray,
        tokens: np.ndarray,
        targets: np.ndarray,
        pad_idx: int,
    ) -> float:
        from captioning.scratch_backprop import evaluate_batch

        return evaluate_batch(self, feature_batch, tokens, targets, pad_idx)

    def backward_batch(
        self,
        feature_batch: np.ndarray,
        tokens: np.ndarray,
        targets: np.ndarray,
        pad_idx: int,
    ):
        return self.loss_and_gradients_batch(feature_batch, tokens, targets, pad_idx)

    def train_batch(
        self,
        feature_batch: np.ndarray,
        tokens: np.ndarray,
        targets: np.ndarray,
        pad_idx: int,
        learning_rate: float,
    ):
        from captioning.scratch_backprop import train_batch

        return train_batch(self, feature_batch, tokens, targets, pad_idx, learning_rate)


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
    if h5py is None:
        raise ModuleNotFoundError(
            "h5py is required to load Keras .weights.h5 files. Install dependencies with: pip install -r requirements.txt"
        )
    if experiment_id not in SELECTED_EXPERIMENTS:
        raise ValueError(f"Scratch decoder supports pre-inject experiments only: {sorted(SELECTED_EXPERIMENTS)}")

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


def extract_feature_from_image(image_path: str | Path) -> np.ndarray:
    from common.image_io import load_image
    from captioning.extract_inception_features import build_encoder, inception_preprocess

    image = load_image(image_path, target_size=(299, 299), normalize=False)
    batch = inception_preprocess(image[None, ...])
    encoder = build_encoder()
    feature = encoder.predict(batch, verbose=0)[0]
    return np.asarray(feature, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run selected captioning decoders with NumPy scratch forward propagation.")
    parser.add_argument("--experiment-id", choices=sorted(SELECTED_EXPERIMENTS), default="lstm_layers1_hidden128")
    parser.add_argument("--mode", choices=SCRATCH_MODES, default=PREDICT_MODE)
    parser.add_argument("--image", type=Path, help="Run end-to-end scratch inference from a raw image file.")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--compare-keras", action="store_true", help="Print Keras and scratch greedy captions side by side.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluate/backward/train-batch/fit modes.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="SGD learning rate for scratch training modes.")
    parser.add_argument("--epochs", type=int, default=1, help="Epoch count for fit mode.")
    parser.add_argument("--limit-samples", type=int, help="Limit sample count for fit mode.")
    parser.add_argument("--save-updated-weights", type=Path, help="Save scratch-updated weights to this .npz path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decoder = load_scratch_decoder(args.experiment_id)

    if args.mode != PREDICT_MODE:
        run_training_style_mode(args, decoder)
        return

    keras_model = load_keras_model(args.experiment_id) if args.compare_keras else None

    if args.image is not None:
        feature = extract_feature_from_image(args.image)
        scratch_prediction = decoder.greedy_decode(feature)
        print(f"Scratch raw-image greedy decoding: {args.experiment_id}")
        print(f"Image: {args.image}")
        if keras_model is not None:
            keras_prediction = keras_greedy_decode(keras_model, feature, decoder)
            print(f"Keras:   {keras_prediction}")
            print(f"Scratch: {scratch_prediction}")
            print(f"Match:   {keras_prediction == scratch_prediction}")
        else:
            print(f"Predicted: {scratch_prediction}")
        return

    features, image_id_to_index, captions = load_feature_inputs(args.split)

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


def run_training_style_mode(args: argparse.Namespace, decoder: ScratchDecoder) -> None:
    from captioning.scratch_backprop import (
        default_scratch_weights_path,
        fit_scratch,
        load_teacher_forcing_batch,
        save_scratch_weights,
    )

    if args.mode == "evaluate":
        feature_batch, tokens, targets, pad_idx = load_teacher_forcing_batch(args.split, args.batch_size)
        loss = decoder.evaluate_batch(feature_batch, tokens, targets, pad_idx)
        print(f"Scratch evaluate: {args.experiment_id} ({args.split})")
        print(f"Batch size: {len(feature_batch)}")
        print(f"Loss: {loss:.6f}")
        return

    if args.mode == "backward":
        feature_batch, tokens, targets, pad_idx = load_teacher_forcing_batch(args.split, args.batch_size)
        loss, gradients = decoder.backward_batch(feature_batch, tokens, targets, pad_idx)
        print(f"Scratch backward propagation: {args.experiment_id} ({args.split})")
        print(f"Batch size: {len(feature_batch)}")
        print(f"Loss: {loss:.6f}")
        print_gradient_summary(gradients)
        return

    if args.mode == "train-batch":
        feature_batch, tokens, targets, pad_idx = load_teacher_forcing_batch(args.split, args.batch_size)
        before_loss = decoder.evaluate_batch(feature_batch, tokens, targets, pad_idx)
        loss, updated_decoder, gradients = decoder.train_batch(
            feature_batch,
            tokens,
            targets,
            pad_idx,
            args.learning_rate,
        )
        after_loss = updated_decoder.evaluate_batch(feature_batch, tokens, targets, pad_idx)
        output_path = args.save_updated_weights or default_scratch_weights_path(args.experiment_id)
        saved_path = save_scratch_weights(updated_decoder, output_path)
        print(f"Scratch train-batch: {args.experiment_id} ({args.split})")
        print(f"Batch size: {len(feature_batch)}")
        print(f"Learning rate: {args.learning_rate}")
        print(f"Loss before update: {before_loss:.6f}")
        print(f"Backward loss: {loss:.6f}")
        print(f"Loss after one SGD update on same batch: {after_loss:.6f}")
        print_gradient_summary(gradients)
        print(f"Saved scratch-updated weights: {saved_path}")
        return

    if args.mode == "fit":
        updated_decoder, history = fit_scratch(
            decoder=decoder,
            split=args.split,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            limit_samples=args.limit_samples,
        )
        output_path = args.save_updated_weights or default_scratch_weights_path(args.experiment_id)
        saved_path = save_scratch_weights(updated_decoder, output_path)
        print(f"Scratch fit: {args.experiment_id} ({args.split})")
        print(f"Batch size: {args.batch_size}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.learning_rate}")
        if args.limit_samples is not None:
            print(f"Limit samples: {args.limit_samples}")
        if history:
            print(f"Initial epoch loss: {history[0]['loss']:.6f}")
            print(f"Final epoch loss: {history[-1]['loss']:.6f}")
        print(f"Saved scratch-updated weights: {saved_path}")
        return

    raise ValueError(f"Unsupported mode: {args.mode}")


def print_gradient_summary(gradients) -> None:
    print("\nGradient shapes and L2 norms:")
    print_array_summary("image_projection.kernel", gradients.image_projection.kernel)
    print_array_summary("image_projection.bias", gradients.image_projection.bias)
    print_array_summary("token_embedding", gradients.token_embedding)
    for index, grad in enumerate(gradients.recurrent_layers):
        print_array_summary(f"recurrent_layers.{index}.kernel", grad.kernel)
        print_array_summary(f"recurrent_layers.{index}.recurrent_kernel", grad.recurrent_kernel)
        print_array_summary(f"recurrent_layers.{index}.bias", grad.bias)
    print_array_summary("token_distribution.kernel", gradients.token_distribution.kernel)
    print_array_summary("token_distribution.bias", gradients.token_distribution.bias)


def print_array_summary(name: str, array: np.ndarray) -> None:
    print(f"{name}: shape={array.shape}, norm={np.linalg.norm(array):.6f}")


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
