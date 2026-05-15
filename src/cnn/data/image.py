from pathlib import Path

import numpy as np
from PIL import Image

def iter_image_batches(image_paths, target_size, batch_size, normalize=True, dtype=np.float32):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    pending_paths = []
    for image_path in image_paths:
        pending_paths.append(image_path)
        if len(pending_paths) == batch_size:
            yield load_image_batch(
                image_paths=pending_paths,
                target_size=target_size,
                normalize=normalize,
                dtype=dtype,
            )
            pending_paths = []

    if pending_paths:
        yield load_image_batch(
            image_paths=pending_paths,
            target_size=target_size,
            normalize=normalize,
            dtype=dtype,
        )

def load_image(image_path, target_size, normalize=True, dtype=np.float32):
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

def load_image_batch(image_paths, target_size, normalize=True, dtype=np.float32):
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
        return np.empty((0, height, width, 3), dtype=dtype)

    return np.stack(arrays, axis=0)

def extract_features(
    image_paths,
    encoder,
    target_size,
    batch_size=32,
    normalize=True,
    preprocess_fn=None,
    verbose=0,
):
    feature_batches = []

    for batch in iter_image_batches(
        image_paths=image_paths,
        target_size=target_size,
        batch_size=batch_size,
        normalize=normalize,
    ):
        if preprocess_fn is not None:
            batch = preprocess_fn(batch)

        features = encoder.predict(batch, verbose=verbose)
        feature_batches.append(np.asarray(features))

    if not feature_batches:
        return np.empty((0,), dtype=np.float32)

    return np.concatenate(feature_batches, axis=0)

def save_features(features, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features)

def extract_and_save_features(
    image_paths,
    encoder,
    target_size,
    output_path,
    batch_size=32,
    normalize=True,
    preprocess_fn=None,
    verbose=0,
):
    features = extract_features(
        image_paths,
        encoder,
        target_size,
        batch_size,
        normalize,
        preprocess_fn,
        verbose,
    )
    save_features(features, output_path)
    return features