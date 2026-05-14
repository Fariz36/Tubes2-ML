from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from captioning.fafo.training_results import (
    EVAL_BATCH_SIZE,
    FAFO_OUTPUT_DIR,
    IMAGES_DIR,
    MODEL_DIR,
    build_model_from_summary,
    find_summary_row,
    greedy_decode_batch,
    load_inference_inputs,
    load_summary,
    resolve_artifact_path,
    score_caption,
)


def generate_qualitative_examples(
    rnn_experiment_id: str,
    lstm_experiment_id: str,
    split: str,
    count: int | None,
    output_count: int,
) -> tuple[Path, Path]:
    summary = load_summary()
    rnn_model = load_model(rnn_experiment_id, summary)
    lstm_model = load_model(lstm_experiment_id, summary)
    features, image_id_to_index, captions, word_to_idx, idx_to_word, max_steps = load_inference_inputs(split)
    selected_image_ids = list(captions)[:count]
    if not selected_image_ids:
        raise ValueError(f"No images selected for split={split!r} and count={count!r}")

    rows = []
    for start in tqdm(range(0, len(selected_image_ids), EVAL_BATCH_SIZE), desc="Generating qualitative rows", unit="batch"):
        batch_image_ids = selected_image_ids[start:start + EVAL_BATCH_SIZE]
        feature_indices = [image_id_to_index[image_id] for image_id in batch_image_ids]
        feature_batch = features[feature_indices]
        rnn_predictions = greedy_decode_batch(rnn_model, feature_batch, word_to_idx, idx_to_word, max_steps)
        lstm_predictions = greedy_decode_batch(lstm_model, feature_batch, word_to_idx, idx_to_word, max_steps)

        for image_id, rnn_prediction, lstm_prediction in zip(batch_image_ids, rnn_predictions, lstm_predictions, strict=True):
            ground_truths = [str(caption) for caption in captions[image_id]]
            rnn_bleu4, rnn_meteor = score_caption(rnn_prediction, ground_truths)
            lstm_bleu4, lstm_meteor = score_caption(lstm_prediction, ground_truths)
            rows.append(
                {
                    "image_id": image_id,
                    "rnn_experiment_id": rnn_experiment_id,
                    "lstm_experiment_id": lstm_experiment_id,
                    "rnn_prediction": rnn_prediction,
                    "lstm_prediction": lstm_prediction,
                    "rnn_bleu4": rnn_bleu4,
                    "rnn_meteor": rnn_meteor,
                    "lstm_bleu4": lstm_bleu4,
                    "lstm_meteor": lstm_meteor,
                    "mean_bleu4": (rnn_bleu4 + lstm_bleu4) / 2,
                    "ground_truths": " | ".join(ground_truths),
                }
            )

    examples = choose_examples(rows, output_count)
    FAFO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = FAFO_OUTPUT_DIR / "qualitative_20_examples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(examples[0]))
        writer.writeheader()
        writer.writerows(examples)

    png_path = FAFO_OUTPUT_DIR / "qualitative_20_examples.png"
    save_examples_plot(examples, png_path)
    print_summary(examples)
    return csv_path, png_path


def load_model(experiment_id: str, summary: list[dict[str, object]]):
    row = find_summary_row(summary, experiment_id)
    model = build_model_from_summary(row)
    weights_path = resolve_artifact_path(str(row["weights_file"]))
    if not weights_path.exists():
        weights_path = MODEL_DIR / f"{experiment_id}.weights.h5"
    model.load_weights(weights_path)
    return model


def choose_examples(rows: list[dict[str, object]], output_count: int) -> list[dict[str, object]]:
    if output_count < 3:
        raise ValueError("output_count must be at least 3")
    ranked = sorted(rows, key=lambda row: float(row["mean_bleu4"]), reverse=True)
    bucket_sizes = split_bucket_sizes(output_count)
    high = ranked[:bucket_sizes[0]]
    middle_start = max(0, (len(ranked) - bucket_sizes[1]) // 2)
    medium = ranked[middle_start:middle_start + bucket_sizes[1]]
    low = list(reversed(ranked[-bucket_sizes[2]:]))

    examples = []
    for bucket, bucket_rows in (("high", high), ("medium", medium), ("low", low)):
        for row in bucket_rows:
            row = dict(row)
            row["quality_bucket"] = bucket
            examples.append(row)
    return examples[:output_count]


def split_bucket_sizes(total: int) -> tuple[int, int, int]:
    base = total // 3
    remainder = total % 3
    return (
        base + int(remainder > 0),
        base + int(remainder > 1),
        base,
    )


def save_examples_plot(examples: list[dict[str, object]], output_path: Path) -> None:
    fig = plt.figure(figsize=(14, max(4, 3.1 * len(examples))))
    for row_index, row in enumerate(examples, start=1):
        image_ax = fig.add_subplot(len(examples), 2, (row_index - 1) * 2 + 1)
        image_ax.set_title(f"{row['quality_bucket']} | {row['image_id']}")
        image_ax.set_xticks([])
        image_ax.set_yticks([])
        image_path = IMAGES_DIR / str(row["image_id"])
        if image_path.exists():
            with Image.open(image_path) as image:
                image_ax.imshow(image.convert("RGB"))
        else:
            image_ax.text(0.5, 0.5, "Missing image", ha="center", va="center")

        text_ax = fig.add_subplot(len(examples), 2, (row_index - 1) * 2 + 2)
        text_ax.axis("off")
        text = (
            f"RNN: {row['rnn_prediction']}\n"
            f"RNN BLEU-4/METEOR: {float(row['rnn_bleu4']):.4f} / {float(row['rnn_meteor']):.4f}\n\n"
            f"LSTM: {row['lstm_prediction']}\n"
            f"LSTM BLEU-4/METEOR: {float(row['lstm_bleu4']):.4f} / {float(row['lstm_meteor']):.4f}\n\n"
            "Ground truth:\n"
            + "\n".join(f"- {caption}" for caption in str(row["ground_truths"]).split(" | "))
        )
        text_ax.text(0, 1, text, va="top", wrap=True, fontsize=9)

    fig.suptitle("Qualitative RNN vs LSTM Caption Examples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_summary(examples: list[dict[str, object]]) -> None:
    print("\nQualitative examples")
    print("bucket | image_id | mean BLEU-4 | RNN BLEU-4 | LSTM BLEU-4")
    print("-------+----------+-------------+------------+------------")
    for row in examples:
        print(
            f"{str(row['quality_bucket']):<6} | "
            f"{str(row['image_id']):<8} | "
            f"{float(row['mean_bleu4']):>11.4f} | "
            f"{float(row['rnn_bleu4']):>10.4f} | "
            f"{float(row['lstm_bleu4']):>10.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate high/medium/low qualitative RNN vs LSTM examples.")
    parser.add_argument("--rnn-experiment-id", default="rnn_layers2_hidden128")
    parser.add_argument("--lstm-experiment-id", default="lstm_layers1_hidden128")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--count", type=int, default=1000, help="Candidate image count. Use 0 for the full split.")
    parser.add_argument("--output-count", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = None if args.count == 0 else args.count
    csv_path, png_path = generate_qualitative_examples(
        rnn_experiment_id=args.rnn_experiment_id,
        lstm_experiment_id=args.lstm_experiment_id,
        split=args.split,
        count=count,
        output_count=args.output_count,
    )
    print(f"Saved qualitative examples CSV: {csv_path}")
    print(f"Saved qualitative examples plot: {png_path}")


if __name__ == "__main__":
    main()
