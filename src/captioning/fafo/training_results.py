from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from PIL import Image
from tqdm import tqdm

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
EVAL_BATCH_SIZE = 256
METEOR_READY = False


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


def ensure_meteor_resources() -> None:
    global METEOR_READY
    if METEOR_READY:
        return

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

    try:
        nltk.data.find("corpora/omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

    try:
        meteor_score([["dog"]], ["canine"])
    except LookupError as exc:
        raise RuntimeError(
            "METEOR needs NLTK WordNet data. Run: python -m nltk.downloader wordnet omw-1.4"
        ) from exc

    METEOR_READY = True


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


def greedy_decode_batch(
    model,
    feature_batch: np.ndarray,
    word_to_idx: dict[str, int],
    idx_to_word: dict[str, str],
    max_steps: int,
) -> list[str]:
    pad_idx = word_to_idx["<pad>"]
    start_idx = word_to_idx["<start>"]
    end_idx = word_to_idx["<end>"]
    batch_size = feature_batch.shape[0]
    tokens = np.full((batch_size, max_steps), pad_idx, dtype=np.int32)
    tokens[:, 0] = start_idx
    generated: list[list[str]] = [[] for _ in range(batch_size)]
    finished = np.zeros((batch_size,), dtype=bool)

    for step in range(max_steps):
        predictions = model.predict([feature_batch, tokens], batch_size=batch_size, verbose=0)
        next_indices = np.argmax(predictions[:, step, :], axis=1).astype(np.int32)

        for row_index, next_idx in enumerate(next_indices):
            if finished[row_index]:
                continue
            if next_idx == end_idx:
                finished[row_index] = True
                continue

            word = idx_to_word.get(str(int(next_idx)), "<unk>")
            if word not in {"<pad>", "<start>", "<unk>"}:
                generated[row_index].append(word)
            if step + 1 < max_steps:
                tokens[row_index, step + 1] = next_idx

        if finished.all():
            break

    return [" ".join(words) if words else "<empty>" for words in generated]


def score_caption(prediction: str, ground_truths: list[str]) -> tuple[float, float]:
    ensure_meteor_resources()
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
    meteor = meteor_score(reference_tokens, predicted_tokens)
    return float(bleu4), float(meteor)


def load_inference_inputs(split: str) -> tuple[np.ndarray, dict[str, int], dict[str, list[str]], dict[str, int], dict[str, str], int]:
    features = np.load(FEATURE_DIR / f"{FEATURE_BASENAME}_features.npy").astype("float32", copy=False)
    image_ids = load_json(FEATURE_DIR / f"{FEATURE_BASENAME}_image_ids.json")
    image_id_to_index = {image_id: index for index, image_id in enumerate(image_ids)}
    captions = load_json(SPLITS_DIR / f"{split}_captions.json")
    word_to_idx = {str(word): int(index) for word, index in load_json(PREPROCESSED_DIR / "word_to_idx.json").items()}
    idx_to_word = {str(index): str(word) for index, word in load_json(PREPROCESSED_DIR / "idx_to_word.json").items()}
    max_steps = int(load_json(TEACHER_FORCING_DIR / "teacher_forcing_metadata.json")["decoder_timesteps"])
    return features, image_id_to_index, captions, word_to_idx, idx_to_word, max_steps


def show_inference_examples(
    examples: list[tuple[str, str, list[str], float, float]],
    experiment_id: str,
    split: str,
    show: bool,
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
        caption_text += f"\n\nScores:\nBLEU-4: {bleu4:.4f}\nMETEOR: {meteor:.4f}"
        text_ax.text(0, 1, caption_text, va="top", wrap=True, fontsize=10)

    fig.suptitle(f"Greedy Inference Examples: {experiment_id} ({split})")
    fig.tight_layout()

    FAFO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FAFO_OUTPUT_DIR / f"{experiment_id}_{split}_inference_examples.png"
    fig.savefig(output_path, dpi=160)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def run_inference(summary: list[dict[str, object]], experiment_id: str, split: str, count: int, show: bool) -> None:
    row = find_summary_row(summary, experiment_id)
    model = build_model_from_summary(row)
    weights_path = resolve_artifact_path(str(row["weights_file"]))
    if not weights_path.exists():
        weights_path = MODEL_DIR / f"{experiment_id}.weights.h5"
    model.load_weights(weights_path)

    features, image_id_to_index, captions, word_to_idx, idx_to_word, max_steps = load_inference_inputs(split)

    print(f"\Inference with {experiment_id} on {split} split")
    examples: list[tuple[str, str, list[str], float, float]] = []
    for image_id in list(captions)[:count]:
        feature = features[image_id_to_index[image_id]]
        prediction = greedy_decode(model, feature, word_to_idx, idx_to_word, max_steps)
        ground_truths = [str(caption) for caption in captions[image_id]]
        bleu4, meteor = score_caption(prediction, ground_truths)
        examples.append((image_id, prediction, ground_truths, bleu4, meteor))
        print(f"\nImage: {image_id}")
        print(f"Predicted: {prediction}")
        print(f"Scores: BLEU-4={bleu4:.4f}, METEOR={meteor:.4f}")
        print("Ground truth:")
        for caption in ground_truths:
            print(f"- {caption}")

    bleu_scores = [example[3] for example in examples]
    meteor_scores = [example[4] for example in examples if example[4] is not None]
    print("\nSample inference score summary")
    print(f"Mean BLEU-4: {sum(bleu_scores) / len(bleu_scores):.4f}")
    print(f"Mean METEOR: {sum(meteor_scores) / len(meteor_scores):.4f}")

    output_path = show_inference_examples(examples, experiment_id, split, show)
    print(f"Saved inference examples plot: {output_path}")


def evaluate_experiment(
    row: dict[str, object],
    split: str,
    eval_count: int | None,
) -> dict[str, object]:
    model = build_model_from_summary(row)
    weights_path = resolve_artifact_path(str(row["weights_file"]))
    if not weights_path.exists():
        weights_path = MODEL_DIR / f"{row['experiment_id']}.weights.h5"
    model.load_weights(weights_path)

    features, image_id_to_index, captions, word_to_idx, idx_to_word, max_steps = load_inference_inputs(split)
    selected_image_ids = list(captions)[:eval_count]
    if not selected_image_ids:
        raise ValueError(f"No images selected for split={split!r} and eval_count={eval_count!r}")

    warmup_feature = features[image_id_to_index[selected_image_ids[0]]]
    greedy_decode_batch(model, warmup_feature[None, :], word_to_idx, idx_to_word, max_steps)

    start_time = time.perf_counter()
    bleu_scores: list[float] = []
    meteor_scores: list[float] = []
    for start in tqdm(range(0, len(selected_image_ids), EVAL_BATCH_SIZE), desc=str(row["experiment_id"]), unit="batch", leave=False):
        batch_image_ids = selected_image_ids[start:start + EVAL_BATCH_SIZE]
        feature_indices = [image_id_to_index[image_id] for image_id in batch_image_ids]
        predictions = greedy_decode_batch(model, features[feature_indices], word_to_idx, idx_to_word, max_steps)
        for image_id, prediction in zip(batch_image_ids, predictions, strict=True):
            ground_truths = [str(caption) for caption in captions[image_id]]
            bleu4, meteor = score_caption(prediction, ground_truths)
            bleu_scores.append(bleu4)
            meteor_scores.append(meteor)

    elapsed_seconds = time.perf_counter() - start_time
    return {
        "experiment_id": row["experiment_id"],
        "model_kind": row["model_kind"],
        "recurrent_layers": row["recurrent_layers"],
        "hidden_units": row["hidden_units"],
        "images": len(selected_image_ids),
        "bleu4": sum(bleu_scores) / len(bleu_scores),
        "meteor": sum(meteor_scores) / len(meteor_scores),
        "inference_seconds": elapsed_seconds,
        "seconds_per_image": elapsed_seconds / len(selected_image_ids),
    }


def print_evaluation_table(rows: list[dict[str, object]]) -> None:
    ranked = sorted(rows, key=lambda row: float(row["bleu4"]), reverse=True)
    print("\Evaluation ranking by BLEU-4")
    print("rank | experiment_id              | kind | layers | hidden | images | BLEU-4 | METEOR | seconds | sec/img")
    print("-----+----------------------------+------+--------+--------+--------+--------+--------+---------+--------")
    for rank, row in enumerate(ranked, start=1):
        print(
            f"{rank:>4} | "
            f"{str(row['experiment_id']):<26} | "
            f"{str(row['model_kind']):<4} | "
            f"{int(row['recurrent_layers']):>6} | "
            f"{int(row['hidden_units']):>6} | "
            f"{int(row['images']):>6} | "
            f"{float(row['bleu4']):>6.4f} | "
            f"{float(row['meteor']):>6.4f} | "
            f"{float(row['inference_seconds']):>7.1f} | "
            f"{float(row['seconds_per_image']):>6.3f}"
        )


def run_evaluation_table(summary: list[dict[str, object]], split: str, eval_count: int | None) -> Path:
    selected = sorted(summary, key=lambda row: float(row["final_val_loss"]))
    rows = []
    for row in tqdm(selected, desc="Evaluating models", unit="model"):
        rows.append(evaluate_experiment(row, split, eval_count))

    print_evaluation_table(rows)

    FAFO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "full" if eval_count is None else str(eval_count)
    output_path = FAFO_OUTPUT_DIR / f"{split}_evaluation_{suffix}_images.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test to see the result of keras captioning decoders."
    )
    parser.add_argument("--eval-table", action="store_true", help="Evaluate every trained model and save a BLEU-4/METEOR table.")
    parser.add_argument("--eval-count", type=int, help="Limit evaluation table to the first N images. Defaults to the full split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_summary()

    print_training_summary(summary)
    if args.eval_table:
        table_path = run_evaluation_table(summary, "test", args.eval_count)
        print(f"Saved evaluation table: {table_path}")


if __name__ == "__main__":
    main()
