from __future__ import annotations

import csv
import json
from pathlib import Path


CaptionMap = dict[str, list[str]]


def load_captions(captions_path: str | Path) -> CaptionMap:
    path = Path(captions_path)
    if not path.exists():
        raise FileNotFoundError(f"Captions file not found: {path}")

    captions: CaptionMap = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        expected_columns = {"image", "caption"}
        if not expected_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"Expected columns {expected_columns} in {path}")

        for row in reader:
            image_id = row["image"].strip()
            caption = row["caption"].strip()
            captions.setdefault(image_id, []).append(caption)

    return captions


def list_image_ids(images_dir: str | Path) -> list[str]:
    path = Path(images_dir)
    if not path.exists():
        raise FileNotFoundError(f"Images directory not found: {path}")

    return sorted(image_path.name for image_path in path.glob("*.jpg"))


def make_splits(
    image_ids: list[str],
    train_size: int = 6000,
    val_size: int = 1000,
    test_size: int = 1000,
) -> dict[str, list[str]]:
    required = train_size + val_size + test_size
    if len(image_ids) < required:
        raise ValueError(f"Need at least {required} images, got {len(image_ids)}")

    train_end = train_size
    val_end = train_end + val_size
    test_end = val_end + test_size

    return {
        "train": image_ids[:train_end],
        "val": image_ids[train_end:val_end],
        "test": image_ids[val_end:test_end],
        "unused": image_ids[test_end:],
    }


def subset_captions(captions: CaptionMap, image_ids: list[str]) -> CaptionMap:
    missing = [image_id for image_id in image_ids if image_id not in captions]
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"Missing captions for {len(missing)} images: {preview}")

    return {image_id: captions[image_id] for image_id in image_ids}


def save_json(data: object, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def prepare_flickr8k_splits(
    raw_dir: str | Path,
    output_dir: str | Path,
    train_size: int = 6000,
    val_size: int = 1000,
    test_size: int = 1000,
) -> dict[str, list[str]]:
    raw_path = Path(raw_dir)
    captions = load_captions(raw_path / "captions.txt")
    image_ids = list_image_ids(raw_path / "Images")
    image_ids = [image_id for image_id in image_ids if image_id in captions]
    splits = make_splits(image_ids, train_size, val_size, test_size)

    output_path = Path(output_dir)
    save_json(splits, output_path / "image_splits.json")
    for split_name in ("train", "val", "test"):
        save_json(
            subset_captions(captions, splits[split_name]),
            output_path / f"{split_name}_captions.json",
        )

    if splits["unused"]:
        save_json(
            subset_captions(captions, splits["unused"]),
            output_path / "unused_captions.json",
        )

    return splits
