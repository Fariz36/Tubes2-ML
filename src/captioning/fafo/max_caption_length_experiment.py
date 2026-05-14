from __future__ import annotations

import argparse
import csv
import time

import numpy as np
from tqdm import tqdm

from captioning.fafo.training_results import (
    EVAL_BATCH_SIZE,
    FAFO_OUTPUT_DIR,
    find_summary_row,
    load_inference_inputs,
    load_summary,
    resolve_artifact_path,
    score_caption,
    MODEL_DIR,
)
from captioning.fafo.training_results import build_model_from_summary
from captioning.scratch_decoder import load_scratch_decoder


def run_max_caption_length_experiment(
    experiment_id: str,
    split: str,
    count: int | None,
    max_steps_values: list[int],
    backend: str,
) -> str:
    if backend == "keras":
        summary = load_summary()
        row = find_summary_row(summary, experiment_id)
        decoder = build_model_from_summary(row)
        weights_path = resolve_artifact_path(str(row["weights_file"]))
        if not weights_path.exists():
            weights_path = MODEL_DIR / f"{experiment_id}.weights.h5"
        decoder.load_weights(weights_path)
    elif backend == "scratch":
        decoder = load_scratch_decoder(experiment_id)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    features, image_id_to_index, captions, word_to_idx, idx_to_word, default_max_steps = load_inference_inputs(split)
    selected_image_ids = list(captions)[:count]
    if not selected_image_ids:
        raise ValueError(f"No images selected for split={split!r} and count={count!r}")

    rows = []
    for max_steps in max_steps_values:
        if max_steps < 1 or max_steps > default_max_steps:
            raise ValueError(f"max_steps must be between 1 and {default_max_steps}: {max_steps}")

        started_at = time.perf_counter()
        bleu_scores: list[float] = []
        meteor_scores: list[float] = []
        for start in tqdm(
            range(0, len(selected_image_ids), EVAL_BATCH_SIZE),
            desc=f"max_steps={max_steps}",
            unit="batch",
        ):
            batch_image_ids = selected_image_ids[start:start + EVAL_BATCH_SIZE]
            feature_indices = [image_id_to_index[image_id] for image_id in batch_image_ids]
            feature_batch = features[feature_indices]
            if backend == "keras":
                predictions = keras_greedy_decode_batch_with_limit(
                    decoder,
                    feature_batch,
                    word_to_idx,
                    idx_to_word,
                    model_steps=default_max_steps,
                    generated_steps=max_steps,
                )
            else:
                predictions = scratch_greedy_decode_batch_with_limit(
                    decoder,
                    feature_batch,
                    generated_steps=max_steps,
                )
            for image_id, prediction in zip(batch_image_ids, predictions, strict=True):
                ground_truths = [str(caption) for caption in captions[image_id]]
                bleu4, meteor = score_caption(prediction, ground_truths)
                bleu_scores.append(bleu4)
                meteor_scores.append(meteor)

        elapsed_seconds = time.perf_counter() - started_at
        rows.append(
            {
                "experiment_id": experiment_id,
                "backend": backend,
                "split": split,
                "max_generated_steps": max_steps,
                "images": len(selected_image_ids),
                "bleu4": sum(bleu_scores) / len(bleu_scores),
                "meteor": sum(meteor_scores) / len(meteor_scores),
                "inference_seconds": elapsed_seconds,
                "seconds_per_image": elapsed_seconds / len(selected_image_ids),
            }
        )

    print_table(rows)
    FAFO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "full" if count is None else str(count)
    output_path = FAFO_OUTPUT_DIR / f"{experiment_id}_{backend}_{split}_max_caption_length_{suffix}_images.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return str(output_path)


def print_table(rows: list[dict[str, object]]) -> None:
    print("\nMaximum caption length experiment")
    print("max_steps | images | BLEU-4 | METEOR | seconds | sec/img")
    print("----------+--------+--------+--------+---------+--------")
    for row in rows:
        print(
            f"{int(row['max_generated_steps']):>9} | "
            f"{int(row['images']):>6} | "
            f"{float(row['bleu4']):>6.4f} | "
            f"{float(row['meteor']):>6.4f} | "
            f"{float(row['inference_seconds']):>7.1f} | "
            f"{float(row['seconds_per_image']):>6.3f}"
        )


def keras_greedy_decode_batch_with_limit(
    model,
    feature_batch: np.ndarray,
    word_to_idx: dict[str, int],
    idx_to_word: dict[str, str],
    model_steps: int,
    generated_steps: int,
) -> list[str]:
    pad_idx = word_to_idx["<pad>"]
    start_idx = word_to_idx["<start>"]
    end_idx = word_to_idx["<end>"]
    batch_size = feature_batch.shape[0]
    tokens = np.full((batch_size, model_steps), pad_idx, dtype=np.int32)
    tokens[:, 0] = start_idx
    generated: list[list[str]] = [[] for _ in range(batch_size)]
    finished = np.zeros((batch_size,), dtype=bool)

    for step in range(generated_steps):
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
            if step + 1 < model_steps:
                tokens[row_index, step + 1] = next_idx

        if finished.all():
            break

    return [" ".join(words) if words else "<empty>" for words in generated]


def scratch_greedy_decode_batch_with_limit(decoder, feature_batch: np.ndarray, generated_steps: int) -> list[str]:
    pad_idx = decoder.word_to_idx["<pad>"]
    start_idx = decoder.word_to_idx["<start>"]
    end_idx = decoder.word_to_idx["<end>"]
    batch_size = feature_batch.shape[0]
    tokens = np.full((batch_size, decoder.max_steps), pad_idx, dtype=np.int32)
    tokens[:, 0] = start_idx
    generated: list[list[str]] = [[] for _ in range(batch_size)]
    finished = np.zeros((batch_size,), dtype=bool)

    for step in range(generated_steps):
        predictions = decoder.forward_batch(feature_batch, tokens)
        next_indices = np.argmax(predictions[:, step, :], axis=1).astype(np.int32)

        for row_index, next_idx in enumerate(next_indices):
            if finished[row_index]:
                continue
            if next_idx == end_idx:
                finished[row_index] = True
                continue

            word = decoder.idx_to_word.get(str(int(next_idx)), "<unk>")
            if word not in {"<pad>", "<start>", "<unk>"}:
                generated[row_index].append(word)
            if step + 1 < decoder.max_steps:
                tokens[row_index, step + 1] = next_idx

        if finished.all():
            break

    return [" ".join(words) if words else "<empty>" for words in generated]


def parse_max_steps(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated maximum caption length variations.")
    parser.add_argument("--experiment-id", default="lstm_layers1_hidden128")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--count", type=int, default=200, help="Number of images to evaluate. Use 0 for the full split.")
    parser.add_argument("--max-steps", default="20,30,37", help="Comma-separated generated timestep limits.")
    parser.add_argument("--backend", choices=("keras", "scratch"), default="keras")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = None if args.count == 0 else args.count
    output_path = run_max_caption_length_experiment(
        experiment_id=args.experiment_id,
        split=args.split,
        count=count,
        max_steps_values=parse_max_steps(args.max_steps),
        backend=args.backend,
    )
    print(f"Saved max-caption-length CSV: {output_path}")


if __name__ == "__main__":
    main()
