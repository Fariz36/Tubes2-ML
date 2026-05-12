from __future__ import annotations

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from captioning.fafo.training_results import EVAL_BATCH_SIZE, FAFO_OUTPUT_DIR, greedy_decode_batch, score_caption
from captioning.scratch_decoder import (
    SELECTED_EXPERIMENTS,
    load_feature_inputs,
    load_keras_model,
    load_scratch_decoder,
)


def compare_inference(experiment_id: str, split: str, count: int | None) -> Path:
    scratch_decoder = load_scratch_decoder(experiment_id)
    keras_model = load_keras_model(experiment_id)
    features, image_id_to_index, captions = load_feature_inputs(split)
    selected_image_ids = list(captions)[:count]
    if not selected_image_ids:
        raise ValueError(f"No images selected for split={split!r} and count={count!r}")

    rows = []
    for start in tqdm(range(0, len(selected_image_ids), EVAL_BATCH_SIZE), desc=f"Comparing {experiment_id}", unit="batch"):
        batch_image_ids = selected_image_ids[start:start + EVAL_BATCH_SIZE]
        feature_indices = [image_id_to_index[image_id] for image_id in batch_image_ids]
        feature_batch = features[feature_indices]
        keras_predictions = greedy_decode_batch(
            keras_model,
            feature_batch,
            scratch_decoder.word_to_idx,
            scratch_decoder.idx_to_word,
            scratch_decoder.max_steps,
        )
        scratch_predictions = scratch_decoder.greedy_decode_batch(feature_batch)

        for image_id, keras_prediction, scratch_prediction in zip(batch_image_ids, keras_predictions, scratch_predictions, strict=True):
            ground_truths = [str(caption) for caption in captions[image_id]]
            keras_bleu4, keras_meteor = score_caption(keras_prediction, ground_truths)
            scratch_bleu4, scratch_meteor = score_caption(scratch_prediction, ground_truths)

            row = {
                "image_id": image_id,
                "keras_prediction": keras_prediction,
                "scratch_prediction": scratch_prediction,
                "match": keras_prediction == scratch_prediction,
                "keras_bleu4": keras_bleu4,
                "keras_meteor": keras_meteor,
                "scratch_bleu4": scratch_bleu4,
                "scratch_meteor": scratch_meteor,
                "ground_truths": " | ".join(ground_truths),
            }
            rows.append(row)
            print_example(row, ground_truths)

    print_summary(rows)

    FAFO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "full" if count is None else str(count)
    output_path = FAFO_OUTPUT_DIR / f"{experiment_id}_{split}_keras_vs_scratch_{suffix}_images.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def print_example(row: dict[str, object], ground_truths: list[str]) -> None:
    print(f"\nImage: {row['image_id']}")
    print(f"Keras:   {row['keras_prediction']}")
    print(f"Scratch: {row['scratch_prediction']}")
    print(f"Match:   {row['match']}")
    print(
        f"Keras scores:   BLEU-4={float(row['keras_bleu4']):.4f}, "
        f"METEOR={float(row['keras_meteor']):.4f}"
    )
    print(
        f"Scratch scores: BLEU-4={float(row['scratch_bleu4']):.4f}, "
        f"METEOR={float(row['scratch_meteor']):.4f}"
    )
    print("Ground truth:")
    for caption in ground_truths:
        print(f"- {caption}")


def print_summary(rows: list[dict[str, object]]) -> None:
    match_count = sum(1 for row in rows if row["match"])
    print("\nKeras vs scratch inference summary")
    print(f"Images: {len(rows)}")
    print(f"Exact sentence matches: {match_count}/{len(rows)}")
    print(f"Mean Keras BLEU-4: {mean(rows, 'keras_bleu4'):.4f}")
    print(f"Mean Keras METEOR: {mean(rows, 'keras_meteor'):.4f}")
    print(f"Mean Scratch BLEU-4: {mean(rows, 'scratch_bleu4'):.4f}")
    print(f"Mean Scratch METEOR: {mean(rows, 'scratch_meteor'):.4f}")


def mean(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Keras decoder inference against NumPy scratch forward inference."
    )
    parser.add_argument("--experiment-id", choices=sorted(SELECTED_EXPERIMENTS), default="lstm_layers1_hidden128")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--count", type=int, help="Limit comparison to the first N images. Defaults to the full split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = compare_inference(args.experiment_id, args.split, args.count)
    print(f"Saved comparison CSV: {output_path}")


if __name__ == "__main__":
    main()
