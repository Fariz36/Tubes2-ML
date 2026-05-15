import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_ROOTS = (
    REPO_ROOT / "data" / "intel",
    REPO_ROOT / "data" / "raw" / "intel",
)

@dataclass(slots=True)
class ClassificationSplit:
    image_paths: list
    labels: np.ndarray
    class_names: list
    name: str

    @property
    def size(self):
        return len(self.image_paths)

@dataclass(slots=True)
class ClassificationDataset:
    train: ClassificationSplit
    val: ClassificationSplit
    test: ClassificationSplit

    @property
    def class_names(self):
        return self.train.class_names

    @property
    def num_classes(self):
        return len(self.class_names)

def resolve_dataset_root(dataset_root=None, default_candidates=None):
    if dataset_root is not None:
        candidate = Path(dataset_root)
        if candidate.exists():
            return candidate

        repo_relative = REPO_ROOT / candidate
        if repo_relative.exists():
            return repo_relative

        raise FileNotFoundError(f"Dataset root not found: {candidate}")

    candidates = tuple(default_candidates or DEFAULT_DATASET_ROOTS)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Dataset root not found. Expected one of: "
        f"{searched}"
    )

def _is_classification_split_dir(directory):
    if directory is None or not directory.exists() or not directory.is_dir():
        return False

    for child in directory.iterdir():
        if not child.is_dir():
            continue
        if _sorted_image_paths(child):
            return True
    return False

def _find_split_directory(root, candidates):
    for candidate in candidates:
        path = root / candidate
        if _is_classification_split_dir(path):
            return path
    return None

def _sorted_image_paths(directory):
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    image_paths = []
    for pattern in extensions:
        image_paths.extend(directory.glob(pattern))
    return sorted(image_paths)

def _load_split_directory(split_dir, class_names=None, split_name="split"):
    if class_names is None:
        class_names = sorted(path.name for path in split_dir.iterdir() if path.is_dir())

    image_paths = []
    labels = []
    for label, class_name in enumerate(class_names):
        class_dir = split_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        class_image_paths = _sorted_image_paths(class_dir)
        image_paths.extend(class_image_paths)
        labels.extend([label] * len(class_image_paths))

    return ClassificationSplit(
        image_paths=image_paths,
        labels=np.asarray(labels, dtype=np.int64),
        class_names=list(class_names),
        name=split_name,
    )

def _make_split_subset(train_split, subset_indices, name):
    return ClassificationSplit(
        image_paths=[train_split.image_paths[index] for index in subset_indices],
        labels=train_split.labels[subset_indices],
        class_names=train_split.class_names,
        name=name,
    )

def _split_training_set(train_split, validation_split, random_state):
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_split,
        random_state=random_state,
    )
    indices = np.arange(train_split.size)
    train_indices, val_indices = next(splitter.split(indices, train_split.labels))
    train_subset = _make_split_subset(train_split, train_indices, "train")
    val_subset = _make_split_subset(train_split, val_indices, "val")
    return train_subset, val_subset

def _load_tf_example(path, label, height, width, normalize, preprocess_fn):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image = tf.image.resize(image, [height, width], method="bilinear")
    image = tf.cast(image, tf.float32)
    if normalize:
        image = image / 255.0
    if preprocess_fn is not None:
        image = preprocess_fn(image)
    return image, label

def _serialize_split(split):
    return {
        "name": split.name,
        "size": split.size,
        "class_names": split.class_names,
        "image_paths": [str(image_path) for image_path in split.image_paths],
        "labels": split.labels.tolist(),
    }

def load_image_classification_dataset(
    dataset_root=None,
    validation_split=0.2,
    random_state=42,
    train_candidates=("train", "seg_train/seg_train", "seg_train"),
    val_candidates=("val", "validation", "seg_val/seg_val", "seg_val", "seg_validation"),
    test_candidates=("test", "seg_test/seg_test", "seg_test"),
    default_root_candidates=None,
):
    root = resolve_dataset_root(dataset_root, default_candidates=default_root_candidates)

    train_dir = _find_split_directory(
        root,
        train_candidates,
    )
    val_dir = _find_split_directory(
        root,
        val_candidates,
    )
    test_dir = _find_split_directory(
        root,
        test_candidates,
    )

    if train_dir is None or test_dir is None:
        raise FileNotFoundError(
            "Expected image classification dataset directories for train and test splits. "
            "Supported names include train/seg_train and test/seg_test, "
            "including nested Kaggle-style folders like seg_train/seg_train."
        )

    base_train_split = _load_split_directory(train_dir, split_name="train")
    if val_dir is not None:
        val_split = _load_split_directory(
            val_dir,
            class_names=base_train_split.class_names,
            split_name="val",
        )
        train_split = base_train_split
    else:
        train_split, val_split = _split_training_set(
            train_split=base_train_split,
            validation_split=validation_split,
            random_state=random_state,
        )

    test_split = _load_split_directory(
        test_dir,
        class_names=base_train_split.class_names,
        split_name="test",
    )

    return ClassificationDataset(train=train_split, val=val_split, test=test_split)

def build_tf_dataset(
    split,
    target_size,
    batch_size=32,
    shuffle=False,
    shuffle_buffer_size=None,
    normalize=True,
    preprocess_fn=None,
    cache=False,
    prefetch=True,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    height, width = target_size
    path_ds = tf.data.Dataset.from_tensor_slices(
        ([str(path) for path in split.image_paths], split.labels.astype(np.int32))
    )
    load_example = partial(
        _load_tf_example,
        height=height,
        width=width,
        normalize=normalize,
        preprocess_fn=preprocess_fn,
    )
    dataset = path_ds.map(load_example, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        if shuffle_buffer_size is None:
            shuffle_buffer_size = min(split.size, max(2048, batch_size * 64))
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
    if cache:
        if isinstance(cache, str):
            dataset = dataset.cache(cache)
        else:
            dataset = dataset.cache()
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE if prefetch else 1)
    return dataset

def save_dataset_manifest(dataset, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "train": _serialize_split(dataset.train),
        "val": _serialize_split(dataset.val),
        "test": _serialize_split(dataset.test),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")