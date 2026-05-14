from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
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
    print_mismatches(rows, features, image_id_to_index, keras_model, scratch_decoder)

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


def print_mismatches(rows: list[dict[str, object]], features, image_id_to_index, keras_model, scratch_decoder) -> None:
    mismatches = [row for row in rows if not row["match"]]
    if not mismatches:
        print("\nNo Keras/scratch sentence mismatches found.")
        return

    print("\nKeras/scratch sentence mismatches")
    for row in mismatches:
        print(f"\nImage: {row['image_id']}")
        print(f"Keras:   {row['keras_prediction']}")
        print(f"Scratch: {row['scratch_prediction']}")
        print(
            f"Keras scores:   BLEU-4={float(row['keras_bleu4']):.4f}, "
            f"METEOR={float(row['keras_meteor']):.4f}"
        )
        print(
            f"Scratch scores: BLEU-4={float(row['scratch_bleu4']):.4f}, "
            f"METEOR={float(row['scratch_meteor']):.4f}"
        )
        print("Ground truth:")
        for caption in str(row["ground_truths"]).split(" | "):
            print(f"- {caption}")
        print_divergence(row, features, image_id_to_index, keras_model, scratch_decoder)


def print_divergence(row: dict[str, object], features, image_id_to_index, keras_model, scratch_decoder) -> None:
    feature = features[image_id_to_index[str(row["image_id"])]]
    diagnostics = find_first_divergence(keras_model, scratch_decoder, feature)
    if diagnostics is None:
        sentence_diff = first_sentence_difference(
            str(row["keras_prediction"]),
            str(row["scratch_prediction"]),
        )
        print("Forward divergence: not reproduced in single-image diagnostic rerun")
        if sentence_diff is not None:
            print(f"First differing sentence token position: {sentence_diff['position']}")
            print(f"Keras sentence token:   {sentence_diff['keras_token']}")
            print(f"Scratch sentence token: {sentence_diff['scratch_token']}")
        return

    print(f"First differing token step: {diagnostics['step']}")
    print(f"Keras chose:   {diagnostics['keras_token']} ({diagnostics['keras_probability']:.8f})")
    print(f"Scratch chose: {diagnostics['scratch_token']} ({diagnostics['scratch_probability']:.8f})")
    print("Keras top-5:")
    for token, probability in diagnostics["keras_top5"]:
        print(f"- {token}: {probability:.8f}")
    print("Scratch top-5:")
    for token, probability in diagnostics["scratch_top5"]:
        print(f"- {token}: {probability:.8f}")


def find_first_divergence(keras_model, scratch_decoder, feature):
    pad_idx = scratch_decoder.word_to_idx["<pad>"]
    start_idx = scratch_decoder.word_to_idx["<start>"]
    end_idx = scratch_decoder.word_to_idx["<end>"]
    keras_tokens = np.full((scratch_decoder.max_steps,), pad_idx, dtype=np.int32)
    scratch_tokens = np.full((scratch_decoder.max_steps,), pad_idx, dtype=np.int32)
    keras_tokens[0] = start_idx
    scratch_tokens[0] = start_idx

    for step in range(scratch_decoder.max_steps):
        keras_probabilities = keras_model.predict([feature[None, :], keras_tokens[None, :]], verbose=0)[0, step]
        scratch_probabilities = scratch_decoder.forward(feature, scratch_tokens)[step]
        keras_next_idx = int(np.argmax(keras_probabilities))
        scratch_next_idx = int(np.argmax(scratch_probabilities))

        if keras_next_idx != scratch_next_idx:
            return {
                "step": step,
                "keras_token": token_name(scratch_decoder, keras_next_idx),
                "scratch_token": token_name(scratch_decoder, scratch_next_idx),
                "keras_probability": float(keras_probabilities[keras_next_idx]),
                "scratch_probability": float(scratch_probabilities[scratch_next_idx]),
                "keras_top5": top_tokens(scratch_decoder, keras_probabilities),
                "scratch_top5": top_tokens(scratch_decoder, scratch_probabilities),
            }

        if keras_next_idx == end_idx:
            break
        if step + 1 < scratch_decoder.max_steps:
            keras_tokens[step + 1] = keras_next_idx
            scratch_tokens[step + 1] = scratch_next_idx

    return None


def top_tokens(scratch_decoder, probabilities) -> list[tuple[str, float]]:
    indices = np.argsort(probabilities)[-5:][::-1]
    return [(token_name(scratch_decoder, int(index)), float(probabilities[index])) for index in indices]


def token_name(scratch_decoder, index: int) -> str:
    return scratch_decoder.idx_to_word.get(str(index), f"<missing:{index}>")


def first_sentence_difference(keras_prediction: str, scratch_prediction: str) -> dict[str, object] | None:
    keras_tokens = keras_prediction.split()
    scratch_tokens = scratch_prediction.split()
    for position, (keras_token, scratch_token) in enumerate(zip(keras_tokens, scratch_tokens, strict=False)):
        if keras_token != scratch_token:
            return {
                "position": position,
                "keras_token": keras_token,
                "scratch_token": scratch_token,
            }

    if len(keras_tokens) != len(scratch_tokens):
        return {
            "position": min(len(keras_tokens), len(scratch_tokens)),
            "keras_token": "<end>" if len(keras_tokens) < len(scratch_tokens) else keras_tokens[min(len(keras_tokens), len(scratch_tokens))],
            "scratch_token": "<end>" if len(scratch_tokens) < len(keras_tokens) else scratch_tokens[min(len(keras_tokens), len(scratch_tokens))],
        }

    return None


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
