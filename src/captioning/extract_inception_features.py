from __future__ import annotations

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
BATCH_SIZE = 32
LIMIT: int | None = None
PREPARE_ONLY = False


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


def main() -> None:
    splits = prepare_flickr8k_splits(DEFAULT_RAW_DIR, DEFAULT_SPLITS_DIR)
    if PREPARE_ONLY:
        print(f"Saved splits and captions: {DEFAULT_SPLITS_DIR}")
        return

    image_ids = splits["train"] + splits["val"] + splits["test"] + splits["unused"]
    if LIMIT is not None:
        if LIMIT <= 0:
            raise ValueError("LIMIT must be greater than 0")
        image_ids = image_ids[:LIMIT]

    image_paths = [DEFAULT_RAW_DIR / "Images" / image_id for image_id in image_ids]
    encoder = build_encoder()
    features = extract_features(
        image_paths=image_paths,
        encoder=encoder,
        target_size=(299, 299),
        batch_size=BATCH_SIZE,
        normalize=False,
        preprocess_fn=inception_preprocess,
        verbose=0,
    ).astype(np.float32, copy=False)

    suffix = "" if LIMIT is None else f"_limit{LIMIT}"
    features_path = DEFAULT_FEATURES_DIR / f"{FEATURE_BASENAME}{suffix}_features.npy"
    image_ids_path = DEFAULT_FEATURES_DIR / f"{FEATURE_BASENAME}{suffix}_image_ids.json"
    metadata_path = DEFAULT_FEATURES_DIR / f"{FEATURE_BASENAME}{suffix}_metadata.json"

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
