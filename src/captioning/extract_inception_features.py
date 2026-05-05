from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common.image_io import extract_features, save_features
from captioning.dataset import prepare_flickr8k_splits, save_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw" / "flickr8k"
DEFAULT_SPLITS_DIR = REPO_ROOT / "artifacts" / "captioning" / "splits"
DEFAULT_FEATURES_DIR = REPO_ROOT / "artifacts" / "captioning" / "features"
FEATURE_BASENAME = "inception_v3_flickr8k"


def build_encoder():
    try:
        from tensorflow.keras.applications import InceptionV3
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow is required for InceptionV3 feature extraction. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    encoder = InceptionV3(weights="imagenet", include_top=False, pooling="avg")
    encoder.trainable = False
    return encoder


def inception_preprocess(batch: np.ndarray) -> np.ndarray:
    try:
        from tensorflow.keras.applications.inception_v3 import preprocess_input
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow is required for InceptionV3 preprocessing. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    return preprocess_input(batch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Flickr8k splits and extract frozen InceptionV3 features.",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--splits-dir", type=Path, default=DEFAULT_SPLITS_DIR)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional smoke-test limit for feature extraction.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only write split/caption JSON files; skip TensorFlow feature extraction.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    splits = prepare_flickr8k_splits(args.raw_dir, args.splits_dir)
    if args.prepare_only:
        print(f"Saved splits and captions: {args.splits_dir}")
        return

    image_ids = splits["train"] + splits["val"] + splits["test"] + splits["unused"]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than 0")
        image_ids = image_ids[: args.limit]

    image_paths = [args.raw_dir / "Images" / image_id for image_id in image_ids]
    encoder = build_encoder()
    features = extract_features(
        image_paths=image_paths,
        encoder=encoder,
        target_size=(299, 299),
        batch_size=args.batch_size,
        normalize=False,
        preprocess_fn=inception_preprocess,
        verbose=0,
    ).astype(np.float32, copy=False)

    suffix = "" if args.limit is None else f"_limit{args.limit}"
    features_path = args.features_dir / f"{FEATURE_BASENAME}{suffix}_features.npy"
    image_ids_path = args.features_dir / f"{FEATURE_BASENAME}{suffix}_image_ids.json"
    metadata_path = args.features_dir / f"{FEATURE_BASENAME}{suffix}_metadata.json"

    save_features(features, features_path)
    save_json(image_ids, image_ids_path)
    save_json(
        {
            "encoder": "InceptionV3",
            "weights": "imagenet",
            "include_top": False,
            "pooling": "avg",
            "trainable": False,
            "input_size": [299, 299],
            "preprocess_fn": "tensorflow.keras.applications.inception_v3.preprocess_input",
            "feature_shape": list(features.shape),
            "feature_dtype": str(features.dtype),
            "feature_stage": "before_decoder_projection",
            "image_count": len(image_ids),
            "split_counts": {name: len(ids) for name, ids in splits.items()},
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        metadata_path,
    )

    print(f"Saved features: {features_path}")
    print(f"Saved image ids: {image_ids_path}")
    print(f"Saved metadata: {metadata_path}")


if __name__ == "__main__":
    main()
