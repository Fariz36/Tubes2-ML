from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
from captioning.preprocess_captions import clean_caption

EXPERIMENT_DIR = REPO_ROOT / "artifacts" / "captioning" / "experiments"
HISTORY_DIR = REPO_ROOT / "artifacts" / "captioning" / "histories"
MODEL_DIR = REPO_ROOT / "artifacts" / "captioning" / "models"
FEATURE_DIR = REPO_ROOT / "artifacts" / "captioning" / "features"
PREPROCESSED_DIR = REPO_ROOT / "artifacts" / "captioning" / "preprocessed"
TEACHER_FORCING_DIR = REPO_ROOT / "artifacts" / "captioning" / "teacher_forcing"
SPLITS_DIR = REPO_ROOT / "artifacts" / "captioning" / "splits"
FAFO_OUTPUT_DIR = REPO_ROOT / "artifacts" / "captioning" / "fafo"
IMAGES_DIR = REPO_ROOT / "data" / "raw" / "flickr8k" / "Images"

SUMMARY_PATH = EXPERIMENT_DIR / "decoder_training_summary.json"
FEATURE_BASENAME = "inception_v3_flickr8k"


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_artifact_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path

    parts = path_value.as_posix().split("artifacts/") if isinstance(path_value, Path) else str(path_value).replace("\\", "/").split("artifacts/")
    if len(parts) == 2:
        candidate = REPO_ROOT / "artifacts" / parts[1]
        if candidate.exists():
            return candidate

    return path


def load_summary() -> list[dict[str, object]]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Training summary not found: {SUMMARY_PATH}")

    summary = load_json(SUMMARY_PATH)
    if not isinstance(summary, list):
        raise ValueError(f"Expected list in {SUMMARY_PATH}")
    return [dict(row) for row in summary]


def print_training_summary(summary: list[dict[str, object]]) -> None:
    ranked = sorted(summary, key=lambda row: float(row["final_val_loss"]))
    print("Training result ranking by final validation loss")
    print("rank | experiment_id              | kind | layers | hidden | train_loss | val_loss | minutes")
    print("-----+----------------------------+------+--------+--------+------------+----------+--------")
    for rank, row in enumerate(ranked, start=1):
        minutes = float(row["elapsed_seconds"]) / 60
        print(
            f"{rank:>4} | "
            f"{str(row['experiment_id']):<26} | "
            f"{str(row['model_kind']):<4} | "
            f"{int(row['recurrent_layers']):>6} | "
            f"{int(row['hidden_units']):>6} | "
            f"{float(row['final_loss']):>10.4f} | "
            f"{float(row['final_val_loss']):>8.4f} | "
            f"{minutes:>6.1f}"
        )


def load_history(experiment_id: str) -> dict[str, object]:
    path = HISTORY_DIR / f"{experiment_id}_history.json"
    if not path.exists():
        raise FileNotFoundError(f"History not found: {path}")
    history = load_json(path)
    if not isinstance(history, dict):
        raise ValueError(f"Expected history object in {path}")
    return history


def plot_histories(summary: list[dict[str, object]], experiment_id: str | None, show: bool) -> Path:
    selected = [row for row in summary if experiment_id in (None, row["experiment_id"])]
    if not selected:
        raise ValueError(f"No experiment matched: {experiment_id}")

    fig, ax = plt.subplots(figsize=(11, 6))
    for row in selected:
        history = load_history(str(row["experiment_id"]))["history"]
        epochs = range(1, len(history["loss"]) + 1)
        ax.plot(epochs, history["loss"], linestyle="--", alpha=0.55, label=f"{row['experiment_id']} train")
        ax.plot(epochs, history["val_loss"], label=f"{row['experiment_id']} val")

    ax.set_title("Decoder Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Masked sparse categorical crossentropy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncols=2)
    fig.tight_layout()

    FAFO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = f"{experiment_id}_loss.png" if experiment_id else "decoder_training_losses.png"
    output_path = FAFO_OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def best_experiment_id(summary: list[dict[str, object]]) -> str:
    best = min(summary, key=lambda row: float(row["final_val_loss"]))
    return str(best["experiment_id"])


def find_summary_row(summary: list[dict[str, object]], experiment_id: str) -> dict[str, object]:
    for row in summary:
        if row["experiment_id"] == experiment_id:
            return row
    raise ValueError(f"Unknown experiment_id: {experiment_id}")


def build_model_from_summary(row: dict[str, object]):
    from captioning.decoder import build_lstm_decoder, build_simple_rnn_decoder

    preprocessing = load_json(PREPROCESSED_DIR / "caption_preprocessing_metadata.json")
    teacher_forcing = load_json(TEACHER_FORCING_DIR / "teacher_forcing_metadata.json")
    builder = build_simple_rnn_decoder if row["model_kind"] == "rnn" else build_lstm_decoder
    return builder(
        vocab_size=int(preprocessing["vocab_size"]),
        decoder_timesteps=int(teacher_forcing["decoder_timesteps"]),
        pad_idx=int(teacher_forcing["pad_idx"]),
        feature_dim=int(teacher_forcing["feature_shape"][-1]),
        embed_dim=256,
        hidden_units=(int(row["hidden_units"]),) * int(row["recurrent_layers"]),
        compile_model=False,
    )


def greedy_decode(model, feature: np.ndarray, word_to_idx: dict[str, int], idx_to_word: dict[str, str], max_steps: int) -> str:
    pad_idx = word_to_idx["<pad>"]
    start_idx = word_to_idx["<start>"]
    end_idx = word_to_idx["<end>"]
    tokens = np.full((max_steps,), pad_idx, dtype=np.int32)
    tokens[0] = start_idx
    generated: list[str] = []

    for step in range(max_steps):
        predictions = model.predict([feature[None, :], tokens[None, :]], verbose=0)
        next_idx = int(np.argmax(predictions[0, step]))
        if next_idx == end_idx:
            break

        word = idx_to_word.get(str(next_idx), "<unk>")
        if word not in {"<pad>", "<start>", "<unk>"}:
            generated.append(word)
        if step + 1 < max_steps:
            tokens[step + 1] = next_idx

    return " ".join(generated) if generated else "<empty>"


def score_caption(prediction: str, ground_truths: list[str]) -> tuple[float, float | None]:
    predicted_tokens = prediction.split()
    reference_tokens = [clean_caption(caption) for caption in ground_truths]
    if not predicted_tokens:
        return 0.0, 0.0

    smoothing = SmoothingFunction().method1
    bleu4 = sentence_bleu(
        reference_tokens,
        predicted_tokens,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smoothing,
    )
    try:
        meteor = meteor_score(reference_tokens, predicted_tokens)
    except LookupError:
        meteor = None
    return float(bleu4), None if meteor is None else float(meteor)


def show_inference_examples(
    examples: list[tuple[str, str, list[str], float, float | None]],
    experiment_id: str,
    split: str,
) -> Path:
    fig = plt.figure(figsize=(13, max(4, 3.2 * len(examples))))
    for row, (image_id, prediction, ground_truths, bleu4, meteor) in enumerate(examples, start=1):
        image_ax = fig.add_subplot(len(examples), 2, (row - 1) * 2 + 1)
        image_ax.set_title(image_id)
        image_ax.set_xticks([])
        image_ax.set_yticks([])

        image_path = IMAGES_DIR / image_id
        if image_path.exists():
            with Image.open(image_path) as image:
                image_ax.imshow(image.convert("RGB"))
        else:
            image_ax.text(0.5, 0.5, "Missing image file", ha="center", va="center")

        text_ax = fig.add_subplot(len(examples), 2, (row - 1) * 2 + 2)
        text_ax.axis("off")
        caption_text = "Predicted:\n"
        caption_text += f"{prediction}\n\nGround truth:\n"
        caption_text += "\n".join(f"- {caption}" for caption in ground_truths)
        meteor_text = "n/a" if meteor is None else f"{meteor:.4f}"
        caption_text += f"\n\nScores:\nBLEU-4: {bleu4:.4f}\nMETEOR: {meteor_text}"
        text_ax.text(0, 1, caption_text, va="top", wrap=True, fontsize=10)

    fig.suptitle(f"Greedy Inference Examples: {experiment_id} ({split})")
    fig.tight_layout()

    FAFO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FAFO_OUTPUT_DIR / f"{experiment_id}_{split}_inference_examples.png"
    fig.savefig(output_path, dpi=160)
    plt.show()
    return output_path


def run_inference(summary: list[dict[str, object]], experiment_id: str, split: str, count: int) -> None:
    row = find_summary_row(summary, experiment_id)
    model = build_model_from_summary(row)
    weights_path = resolve_artifact_path(str(row["weights_file"]))
    if not weights_path.exists():
        weights_path = MODEL_DIR / f"{experiment_id}.weights.h5"
    model.load_weights(weights_path)

    features = np.load(FEATURE_DIR / f"{FEATURE_BASENAME}_features.npy").astype("float32", copy=False)
    image_ids = load_json(FEATURE_DIR / f"{FEATURE_BASENAME}_image_ids.json")
    image_id_to_index = {image_id: index for index, image_id in enumerate(image_ids)}
    captions = load_json(SPLITS_DIR / f"{split}_captions.json")
    word_to_idx = {str(word): int(index) for word, index in load_json(PREPROCESSED_DIR / "word_to_idx.json").items()}
    idx_to_word = {str(index): str(word) for index, word in load_json(PREPROCESSED_DIR / "idx_to_word.json").items()}
    max_steps = int(load_json(TEACHER_FORCING_DIR / "teacher_forcing_metadata.json")["decoder_timesteps"])

    print(f"\nGreedy inference with {experiment_id} on {split} split")
    examples: list[tuple[str, str, list[str], float, float | None]] = []
    for image_id in list(captions)[:count]:
        feature = features[image_id_to_index[image_id]]
        prediction = greedy_decode(model, feature, word_to_idx, idx_to_word, max_steps)
        ground_truths = [str(caption) for caption in captions[image_id]]
        bleu4, meteor = score_caption(prediction, ground_truths)
        examples.append((image_id, prediction, ground_truths, bleu4, meteor))
        print(f"\nImage: {image_id}")
        print(f"Predicted: {prediction}")
        meteor_text = "n/a" if meteor is None else f"{meteor:.4f}"
        print(f"Scores: BLEU-4={bleu4:.4f}, METEOR={meteor_text}")
        print("Ground truth:")
        for caption in ground_truths:
            print(f"- {caption}")

    bleu_scores = [example[3] for example in examples]
    meteor_scores = [example[4] for example in examples if example[4] is not None]
    print("\nSample inference score summary")
    print(f"Mean BLEU-4: {sum(bleu_scores) / len(bleu_scores):.4f}")
    if meteor_scores:
        print(f"Mean METEOR: {sum(meteor_scores) / len(meteor_scores):.4f}")
    else:
        print("Mean METEOR: n/a; NLTK WordNet data may be missing")

    output_path = show_inference_examples(examples, experiment_id, split)
    print(f"Saved inference examples plot: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FAFO inspection for trained captioning decoders."
    )
    parser.add_argument("--experiment-id", help="Experiment to plot/infer. Defaults to best final validation loss.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Split used for inference examples.")
    parser.add_argument("--count", type=int, default=5, help="Number of image examples for greedy inference.")
    parser.add_argument("--no-plot", action="store_true", help="Skip loss plot generation.")
    parser.add_argument("--no-infer", action="store_true", help="Skip greedy inference examples.")
    parser.add_argument("--show", action="store_true", help="Show matplotlib windows in addition to saving plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary()
    experiment_id = args.experiment_id or best_experiment_id(summary)

    print_training_summary(summary)
    print(f"\nSelected experiment for detailed FAFO: {experiment_id}")
    if not args.no_plot:
        plot_path = plot_histories(summary, args.experiment_id, args.show)
        print(f"Saved loss plot: {plot_path}")
    if not args.no_infer:
        run_inference(summary, experiment_id, args.split, args.count)


if __name__ == "__main__":
    main()
