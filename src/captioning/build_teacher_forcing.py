from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from captioning.dataset import save_json


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_DIR = REPO_ROOT / "artifacts" / "captioning" / "features"
DEFAULT_PREPROCESSED_DIR = REPO_ROOT / "artifacts" / "captioning" / "preprocessed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "captioning" / "teacher_forcing"
FEATURE_BASENAME = "inception_v3_flickr8k"


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def build_split_arrays(
    padded_captions: dict[str, list[list[int]]],
    image_id_to_feature_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_indices: list[int] = []
    caption_inputs: list[list[int]] = []
    caption_targets: list[list[int]] = []

    for image_id, sequences in padded_captions.items():
        if image_id not in image_id_to_feature_idx:
            raise ValueError(f"Missing feature index for image: {image_id}")

        feature_idx = image_id_to_feature_idx[image_id]
        for sequence in sequences:
            if len(sequence) < 2:
                raise ValueError(f"Sequence must contain at least 2 tokens: {image_id}")
            feature_indices.append(feature_idx)
            caption_inputs.append(sequence[:-1])
            caption_targets.append(sequence[1:])

    return (
        np.asarray(feature_indices, dtype=np.int32),
        np.asarray(caption_inputs, dtype=np.int32),
        np.asarray(caption_targets, dtype=np.int32),
    )


def save_split_npz(
    output_path: str | Path,
    feature_indices: np.ndarray,
    caption_inputs: np.ndarray,
    caption_targets: np.ndarray,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        feature_indices=feature_indices,
        caption_inputs=caption_inputs,
        caption_targets=caption_targets,
    )


def main() -> None:
    image_ids = load_json(DEFAULT_FEATURES_DIR / f"{FEATURE_BASENAME}_image_ids.json")
    feature_metadata = load_json(DEFAULT_FEATURES_DIR / f"{FEATURE_BASENAME}_metadata.json")
    preprocessing_metadata = load_json(
        DEFAULT_PREPROCESSED_DIR / "caption_preprocessing_metadata.json"
    )
    image_id_to_feature_idx = {
        image_id: index for index, image_id in enumerate(image_ids)
    }

    split_shapes = {}
    for split_name in ("train", "val", "test"):
        padded_captions = load_json(
            DEFAULT_PREPROCESSED_DIR / f"{split_name}_padded_captions.json"
        )
        feature_indices, caption_inputs, caption_targets = build_split_arrays(
            padded_captions,
            image_id_to_feature_idx,
        )
        save_split_npz(
            DEFAULT_OUTPUT_DIR / f"{split_name}_teacher_forcing.npz",
            feature_indices=feature_indices,
            caption_inputs=caption_inputs,
            caption_targets=caption_targets,
        )
        split_shapes[split_name] = {
            "feature_indices": list(feature_indices.shape),
            "caption_inputs": list(caption_inputs.shape),
            "caption_targets": list(caption_targets.shape),
        }

    save_json(
        {
            "feature_file": str(
                DEFAULT_FEATURES_DIR / f"{FEATURE_BASENAME}_features.npy"
            ),
            "feature_shape": feature_metadata["feature_shape"],
            "feature_stage": feature_metadata["feature_stage"],
            "max_seq_len": preprocessing_metadata["max_seq_len_train"],
            "decoder_timesteps": preprocessing_metadata["max_seq_len_train"] - 1,
            "pad_idx": preprocessing_metadata["pad_idx"],
            "start_idx": preprocessing_metadata["start_idx"],
            "end_idx": preprocessing_metadata["end_idx"],
            "unk_idx": preprocessing_metadata["unk_idx"],
            "array_dtypes": {
                "feature_indices": "int32",
                "caption_inputs": "int32",
                "caption_targets": "int32",
            },
            "split_shapes": split_shapes,
        },
        DEFAULT_OUTPUT_DIR / "teacher_forcing_metadata.json",
    )

    print(f"Saved teacher-forcing artifacts: {DEFAULT_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
