from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

from captioning.preprocess_captions import UNK_TOKEN, clean_caption, tokenize_caption


REPO_ROOT = Path(__file__).resolve().parents[3]
IMAGES_DIR = REPO_ROOT / "data" / "raw" / "flickr8k" / "Images"
SPLITS_DIR = REPO_ROOT / "artifacts" / "captioning" / "splits"
PREPROCESSED_DIR = REPO_ROOT / "artifacts" / "captioning" / "preprocessed"
DEFAULT_PAIR_IMAGE_COUNT = 5
LENGTH_HISTOGRAM_BINS = 30
EDGE_EXAMPLE_COUNT = 5


def load_json(path: str | Path) -> object:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def load_caption_map(path: str | Path) -> dict[str, list[str]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected caption mapping in {path}")

    return {str(image_id): [str(caption) for caption in captions] for image_id, captions in data.items()}


def load_all_captions() -> dict[str, list[str]]:
    captions_by_image: dict[str, list[str]] = {}
    for split_name in ("train", "val", "test", "unused"):
        split_path = SPLITS_DIR / f"{split_name}_captions.json"
        if split_path.exists():
            captions_by_image.update(load_caption_map(split_path))

    if not captions_by_image:
        raise FileNotFoundError(f"No split caption files found in {SPLITS_DIR}")

    return captions_by_image


def format_caption(caption: str, show_original: bool) -> str:
    tokens = tokenize_caption(caption)
    processed = " ".join(tokens) if tokens else "<empty after cleaning>"
    if show_original:
        return f"Original: {caption}\n   Processed: {processed}"

    return processed


def show_image_caption_pairs(
    captions_by_image: dict[str, list[str]],
    image_id: str | None,
    show_original: bool,
) -> None:
    selected_ids = [image_id] if image_id else list(captions_by_image)[:DEFAULT_PAIR_IMAGE_COUNT]
    missing_ids = [selected_id for selected_id in selected_ids if selected_id not in captions_by_image]
    if missing_ids:
        raise ValueError(f"Image ID not found in caption map: {', '.join(missing_ids)}")

    fig = plt.figure(figsize=(12, max(4, 3 * len(selected_ids))))
    for row, image_id in enumerate(selected_ids, start=1):
        image_path = IMAGES_DIR / image_id
        image_ax = fig.add_subplot(len(selected_ids), 2, (row - 1) * 2 + 1)
        image_ax.set_title(image_id)
        image_ax.set_xticks([])
        image_ax.set_yticks([])

        if image_path.exists():
            with Image.open(image_path) as image:
                image_ax.imshow(image.convert("RGB"))
        else:
            image_ax.text(0.5, 0.5, "Missing image file", ha="center", va="center")

        text_ax = fig.add_subplot(len(selected_ids), 2, (row - 1) * 2 + 2)
        text_ax.axis("off")
        text_ax.set_title("Original and post-processed captions" if show_original else "Post-processed captions")
        captions = captions_by_image[image_id]
        caption_text = "\n".join(
            f"{index}. {format_caption(caption, show_original)}"
            for index, caption in enumerate(captions, start=1)
        )
        text_ax.text(0, 1, caption_text, va="top", wrap=True, fontsize=10)

    fig.tight_layout()
    plt.show()


def caption_lengths(captions_by_image: dict[str, list[str]]) -> list[int]:
    return [
        len(clean_caption(caption))
        for captions in captions_by_image.values()
        for caption in captions
    ]


def show_length_distribution(captions_by_image: dict[str, list[str]]) -> None:
    lengths = caption_lengths(captions_by_image)
    if not lengths:
        print("No captions found for length distribution.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=LENGTH_HISTOGRAM_BINS, edgecolor="black")
    ax.axvline(sum(lengths) / len(lengths), color="red", linestyle="--", label="mean")
    ax.set_title("Processed Caption Length Distribution")
    ax.set_xlabel("Word count after cleaning, without <start>/<end>")
    ax.set_ylabel("Number of captions")
    ax.legend()
    fig.tight_layout()
    plt.show()


def print_basic_summary(captions_by_image: dict[str, list[str]]) -> None:
    lengths = caption_lengths(captions_by_image)
    caption_counts = [len(captions) for captions in captions_by_image.values()]
    missing_images = [image_id for image_id in captions_by_image if not (IMAGES_DIR / image_id).exists()]
    empty_captions = [
        (image_id, caption)
        for image_id, captions in captions_by_image.items()
        for caption in captions
        if not clean_caption(caption)
    ]

    print("Caption preprocessing summary")
    print(f"Images in caption map: {len(captions_by_image)}")
    print(f"Captions: {sum(caption_counts)}")
    print(f"Captions per image: min={min(caption_counts)}, max={max(caption_counts)}")
    print(f"Missing image files: {len(missing_images)}")
    print(f"Empty captions after cleaning: {len(empty_captions)}")
    print(
        "Processed word length: "
        f"min={min(lengths)}, max={max(lengths)}, mean={sum(lengths) / len(lengths):.2f}"
    )

    if missing_images:
        print(f"First missing image examples: {', '.join(missing_images[:5])}")
    if empty_captions:
        preview = "; ".join(f"{image_id}: {caption!r}" for image_id, caption in empty_captions[:3])
        print(f"First empty caption examples: {preview}")


def print_length_examples(captions_by_image: dict[str, list[str]]) -> None:
    rows = [
        (len(clean_caption(caption)), image_id, " ".join(clean_caption(caption)))
        for image_id, captions in captions_by_image.items()
        for caption in captions
    ]
    rows.sort(key=lambda item: item[0])

    print("\nShortest processed captions")
    for length, image_id, caption in rows[:EDGE_EXAMPLE_COUNT]:
        print(f"{length:2d} words | {image_id} | {caption}")

    print("\nLongest processed captions")
    for length, image_id, caption in rows[-EDGE_EXAMPLE_COUNT:]:
        print(f"{length:2d} words | {image_id} | {caption}")


def print_oov_summary(captions_by_image: dict[str, list[str]]) -> None:
    vocab_path = PREPROCESSED_DIR / "word_to_idx.json"
    if not vocab_path.exists():
        print(f"\nVocabulary not found, skipping OOV summary: {vocab_path}")
        return

    word_to_idx = load_json(vocab_path)
    if not isinstance(word_to_idx, dict):
        raise ValueError(f"Expected vocabulary mapping in {vocab_path}")

    vocab = set(word_to_idx)
    oov_counts: dict[str, int] = {}
    total_tokens = 0
    for captions in captions_by_image.values():
        for caption in captions:
            for token in clean_caption(caption):
                total_tokens += 1
                if token not in vocab:
                    oov_counts[token] = oov_counts.get(token, 0) + 1

    oov_total = sum(oov_counts.values())
    unk_idx = word_to_idx.get(UNK_TOKEN, "not found")
    print("\nVocabulary/OOV summary")
    print(f"Vocabulary size: {len(word_to_idx)}")
    print(f"<unk> index: {unk_idx}")
    print(f"OOV token occurrences: {oov_total} / {total_tokens}")
    if oov_counts:
        top_oov = sorted(oov_counts.items(), key=lambda item: item[1], reverse=True)[:10]
        print("Most frequent OOV tokens: " + ", ".join(f"{token}={count}" for token, count in top_oov))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Human-facing checks for Flickr8k caption preprocessing."
    )
    parser.add_argument(
        "--image-id",
        help="Show the pairing visualization for one specific image, for example 1000268201_693b08cb0e.jpg.",
    )
    parser.add_argument("--original", action="store_true", help="Show original captions together with post-processed captions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captions_by_image = load_all_captions()

    print_basic_summary(captions_by_image)
    print_length_examples(captions_by_image)
    print_oov_summary(captions_by_image)
    show_image_caption_pairs(captions_by_image, args.image_id, args.original)
    show_length_distribution(captions_by_image)


if __name__ == "__main__":
    main()
