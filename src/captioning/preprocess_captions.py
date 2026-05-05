from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from captioning.dataset import CaptionMap, save_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS_DIR = REPO_ROOT / "artifacts" / "captioning" / "splits"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "captioning" / "preprocessed"
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]


def clean_caption(caption: str) -> list[str]:
    text = caption.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split() if text else []


def tokenize_caption(caption: str) -> list[str]:
    return [START_TOKEN, *clean_caption(caption), END_TOKEN]


def load_caption_map(path: str | Path) -> CaptionMap:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    return {image_id: list(captions) for image_id, captions in data.items()}


def build_vocabulary(train_captions: CaptionMap) -> tuple[dict[str, int], dict[str, str]]:
    counter: Counter[str] = Counter()
    for captions in train_captions.values():
        for caption in captions:
            counter.update(clean_caption(caption))

    tokens = [*SPECIAL_TOKENS, *sorted(counter)]
    word_to_idx = {token: index for index, token in enumerate(tokens)}
    idx_to_word = {str(index): token for token, index in word_to_idx.items()}
    return word_to_idx, idx_to_word


def encode_caption(caption: str, word_to_idx: dict[str, int]) -> list[int]:
    unk_idx = word_to_idx[UNK_TOKEN]
    return [word_to_idx.get(token, unk_idx) for token in tokenize_caption(caption)]


def encode_caption_map(
    captions_by_image: CaptionMap,
    word_to_idx: dict[str, int],
) -> dict[str, list[list[int]]]:
    encoded: dict[str, list[list[int]]] = {}
    for image_id, captions in captions_by_image.items():
        encoded[image_id] = [encode_caption(caption, word_to_idx) for caption in captions]
    return encoded


def pad_sequence(
    sequence: list[int],
    max_len: int,
    pad_idx: int,
    end_idx: int,
) -> tuple[list[int], bool]:
    if len(sequence) > max_len:
        padded = sequence[:max_len]
        padded[-1] = end_idx
        return padded, True

    return [*sequence, *([pad_idx] * (max_len - len(sequence)))], False


def pad_caption_map(
    encoded_by_image: dict[str, list[list[int]]],
    max_len: int,
    pad_idx: int,
    end_idx: int,
) -> tuple[dict[str, list[list[int]]], int]:
    padded: dict[str, list[list[int]]] = {}
    truncated_count = 0
    for image_id, sequences in encoded_by_image.items():
        padded_sequences = []
        for sequence in sequences:
            padded_sequence, was_truncated = pad_sequence(
                sequence,
                max_len=max_len,
                pad_idx=pad_idx,
                end_idx=end_idx,
            )
            padded_sequences.append(padded_sequence)
            truncated_count += int(was_truncated)
        padded[image_id] = padded_sequences

    return padded, truncated_count


def max_caption_length(captions_by_image: CaptionMap) -> int:
    return max(
        len(tokenize_caption(caption))
        for captions in captions_by_image.values()
        for caption in captions
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Flickr8k captions, build train vocabulary, and encode captions.",
    )
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_captions = load_caption_map(args.splits_dir / "train_captions.json")
    word_to_idx, idx_to_word = build_vocabulary(train_captions)
    max_len = max_caption_length(train_captions)
    pad_idx = word_to_idx[PAD_TOKEN]
    end_idx = word_to_idx[END_TOKEN]

    save_json(word_to_idx, args.output_dir / "word_to_idx.json")
    save_json(idx_to_word, args.output_dir / "idx_to_word.json")

    split_counts: dict[str, int] = {}
    truncated_counts: dict[str, int] = {}
    for split_name in ("train", "val", "test"):
        captions = load_caption_map(args.splits_dir / f"{split_name}_captions.json")
        encoded = encode_caption_map(captions, word_to_idx)
        padded, truncated_count = pad_caption_map(
            encoded,
            max_len=max_len,
            pad_idx=pad_idx,
            end_idx=end_idx,
        )
        save_json(encoded, args.output_dir / f"{split_name}_encoded_captions.json")
        save_json(padded, args.output_dir / f"{split_name}_padded_captions.json")
        split_counts[split_name] = sum(len(items) for items in encoded.values())
        truncated_counts[split_name] = truncated_count

    save_json(
        {
            "vocab_size": len(word_to_idx),
            "special_tokens": SPECIAL_TOKENS,
            "pad_idx": word_to_idx[PAD_TOKEN],
            "start_idx": word_to_idx[START_TOKEN],
            "end_idx": word_to_idx[END_TOKEN],
            "unk_idx": word_to_idx[UNK_TOKEN],
            "max_seq_len_train": max_len,
            "sequence_stage": "integer_encoded_and_padded",
            "split_caption_counts": split_counts,
            "truncated_caption_counts": truncated_counts,
        },
        args.output_dir / "caption_preprocessing_metadata.json",
    )

    print(f"Saved caption preprocessing artifacts: {args.output_dir}")
    print(f"Vocabulary size: {len(word_to_idx)}")


if __name__ == "__main__":
    main()
