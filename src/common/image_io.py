from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image


ArrayPreprocessFn = Callable[[np.ndarray], np.ndarray]


def load_image(
    image_path: str | Path,
    target_size: tuple[int, int],
    normalize: bool = True,
    dtype: np.dtype | str = np.float32,
) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    height, width = target_size

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=dtype)

    if normalize:
        array = array / np.array(255.0, dtype=dtype)

    return array.astype(dtype, copy=False)


def load_image_batch(
    image_paths: Iterable[str | Path],
    target_size: tuple[int, int],
    normalize: bool = True,
    dtype: np.dtype | str = np.float32,
) -> np.ndarray:
    arrays = [
        load_image(
            image_path=path,
            target_size=target_size,
            normalize=normalize,
            dtype=dtype,
        )
        for path in image_paths
    ]

    if not arrays:
        height, width = target_size
        channels = 3
        return np.empty((0, height, width, channels), dtype=dtype)

    return np.stack(arrays, axis=0)


def extract_features(
    image_paths: list[str | Path],
    encoder,
    target_size: tuple[int, int],
    batch_size: int = 32,
    normalize: bool = True,
    preprocess_fn: ArrayPreprocessFn | None = None, # placeholder, ntar ini bakal diganti aja ke function buat extract feature di CNN nya
    verbose: int = 0,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    feature_batches: list[np.ndarray] = []

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch = load_image_batch(
            image_paths=batch_paths,
            target_size=target_size,
            normalize=normalize,
        )

        if preprocess_fn is not None:
            batch = preprocess_fn(batch)

        features = encoder.predict(batch, verbose=verbose)
        feature_batches.append(np.asarray(features))

    if not feature_batches:
        return np.empty((0,), dtype=np.float32)

    return np.concatenate(feature_batches, axis=0)


def save_features(features: np.ndarray, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features)
