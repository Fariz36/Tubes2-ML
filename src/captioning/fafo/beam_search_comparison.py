from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from tqdm import tqdm

from captioning.fafo.training_results import FAFO_OUTPUT_DIR, score_caption
from captioning.scratch_decoder import SELECTED_EXPERIMENTS, load_feature_inputs, load_scratch_decoder


def compare_beam_search(experiment_id: str, split: str, count: int | None, beam_width: int) -> Path:
    decoder = load_scratch_decoder(experiment_id)
    features, image_id_to_index, captions = load_feature_inputs(split)
    selected_image_ids = list(captions)[:count]
    if not selected_image_ids:
        raise ValueError(f"No images selected for split={split!r} and count={count!r}")

    rows = []
    greedy_seconds = 0.0
    beam_seconds = 0.0
    for image_id in tqdm(selected_image_ids, desc=f"Beam search {experiment_id}", unit="image"):
        feature = features[image_id_to_index[image_id]]
        ground_truths = [str(caption) for caption in captions[image_id]]

        start_time = time.perf_counter()
        greedy_prediction = decoder.greedy_decode(feature)
        greedy_seconds += time.perf_counter() - start_time

        start_time = time.perf_counter()
        beam_prediction = decoder.beam_search_decode(feature, beam_width=beam_width)
        beam_seconds += time.perf_counter() - start_time

        greedy_bleu4, greedy_meteor = score_caption(greedy_prediction, ground_truths)
        beam_bleu4, beam_meteor = score_caption(beam_prediction, ground_truths)
        rows.append(
            {
                "image_id": image_id,
                "experiment_id": experiment_id,
                "beam_width": beam_width,
                "greedy_prediction": greedy_prediction,
                "beam_prediction": beam_prediction,
                "greedy_bleu4": greedy_bleu4,
                "greedy_meteor": greedy_meteor,
                "beam_bleu4": beam_bleu4,
                "beam_meteor": beam_meteor,
                "ground_truths": " | ".join(ground_truths),
            }
        )

    print_summary(rows, greedy_seconds, beam_seconds)
    FAFO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "full" if count is None else str(count)
    output_path = FAFO_OUTPUT_DIR / f"{experiment_id}_{split}_greedy_vs_beam_k{beam_width}_{suffix}_images.csv"
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def print_summary(rows: list[dict[str, object]], greedy_seconds: float, beam_seconds: float) -> None:
    image_count = len(rows)
    print("\nGreedy vs beam search summary")
    print(f"Images: {image_count}")
    print(f"Mean greedy BLEU-4: {mean(rows, 'greedy_bleu4'):.4f}")
    print(f"Mean greedy METEOR: {mean(rows, 'greedy_meteor'):.4f}")
    print(f"Mean beam BLEU-4:   {mean(rows, 'beam_bleu4'):.4f}")
    print(f"Mean beam METEOR:   {mean(rows, 'beam_meteor'):.4f}")
    print(f"Greedy seconds: {greedy_seconds:.1f} ({greedy_seconds / image_count:.3f} sec/img)")
    print(f"Beam seconds:   {beam_seconds:.1f} ({beam_seconds / image_count:.3f} sec/img)")


def mean(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare scratch greedy decoding against beam search decoding.")
    parser.add_argument("--experiment-id", choices=sorted(SELECTED_EXPERIMENTS), default="lstm_layers1_hidden128")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--count", type=int, default=10, help="Number of images to evaluate. Use 0 for the full split.")
    parser.add_argument("--beam-width", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = None if args.count == 0 else args.count
    output_path = compare_beam_search(args.experiment_id, args.split, count, args.beam_width)
    print(f"Saved beam search comparison CSV: {output_path}")


if __name__ == "__main__":
    main()
