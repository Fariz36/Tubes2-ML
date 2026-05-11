from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from captioning.decoder import build_lstm_decoder, build_simple_rnn_decoder


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PREPROCESSED_DIR = REPO_ROOT / "artifacts" / "captioning" / "preprocessed"
DEFAULT_TEACHER_FORCING_DIR = REPO_ROOT / "artifacts" / "captioning" / "teacher_forcing"
DEFAULT_EXPERIMENT_DIR = REPO_ROOT / "artifacts" / "captioning" / "experiments"
DEFAULT_MODEL_DIR = REPO_ROOT / "artifacts" / "captioning" / "models"
DEFAULT_HISTORY_DIR = REPO_ROOT / "artifacts" / "captioning" / "histories"

MODEL_KIND = "both"  # "both", "rnn", or "lstm"
EXPERIMENT_ID: str | None = None
EPOCHS = 20
BATCH_SIZE = 64
LIMIT_TRAIN_SAMPLES: int | None = None
LIMIT_VAL_SAMPLES: int | None = None
WRITE_CONFIGS_ONLY = False


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    model_kind: str
    recurrent_layers: int
    hidden_units: int
    embed_dim: int = 256
    learning_rate: float = 1e-3

    @property
    def hidden_units_per_layer(self) -> tuple[int, ...]:
        return (self.hidden_units,) * self.recurrent_layers


def default_experiment_configs() -> list[ExperimentConfig]:
    configs: list[ExperimentConfig] = []
    for model_kind in ("rnn", "lstm"):
        for recurrent_layers in (1, 2, 3):
            for hidden_units in (128, 512):
                configs.append(
                    ExperimentConfig(
                        experiment_id=(
                            f"{model_kind}_layers{recurrent_layers}_hidden{hidden_units}"
                        ),
                        model_kind=model_kind,
                        recurrent_layers=recurrent_layers,
                        hidden_units=hidden_units,
                    )
                )
    return configs


def load_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: object, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def save_config_table(
    configs: list[ExperimentConfig],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "experiment_id",
        "model_kind",
        "recurrent_layers",
        "hidden_units",
        "embed_dim",
        "learning_rate",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            writer.writerow(asdict(config))


def load_training_metadata(
    preprocessed_dir: str | Path,
    teacher_forcing_dir: str | Path,
) -> dict[str, object]:
    preprocessing = load_json(Path(preprocessed_dir) / "caption_preprocessing_metadata.json")
    teacher_forcing = load_json(Path(teacher_forcing_dir) / "teacher_forcing_metadata.json")
    return {
        "vocab_size": preprocessing["vocab_size"],
        "pad_idx": teacher_forcing["pad_idx"],
        "decoder_timesteps": teacher_forcing["decoder_timesteps"],
        "feature_dim": teacher_forcing["feature_shape"][-1],
        "feature_file": teacher_forcing["feature_file"],
    }


def load_split_arrays(
    teacher_forcing_dir: str | Path,
    split_name: str,
    features: np.ndarray,
    limit_samples: int | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    path = Path(teacher_forcing_dir) / f"{split_name}_teacher_forcing.npz"
    data = np.load(path)
    feature_indices = data["feature_indices"]
    caption_inputs = data["caption_inputs"]
    caption_targets = data["caption_targets"]

    if limit_samples is not None:
        feature_indices = feature_indices[:limit_samples]
        caption_inputs = caption_inputs[:limit_samples]
        caption_targets = caption_targets[:limit_samples]

    return (features[feature_indices], caption_inputs), caption_targets


def build_model(
    config: ExperimentConfig,
    metadata: dict[str, object],
) -> tf.keras.Model:
    builder = build_simple_rnn_decoder if config.model_kind == "rnn" else build_lstm_decoder
    return builder(
        vocab_size=int(metadata["vocab_size"]),
        decoder_timesteps=int(metadata["decoder_timesteps"]),
        pad_idx=int(metadata["pad_idx"]),
        feature_dim=int(metadata["feature_dim"]),
        embed_dim=config.embed_dim,
        hidden_units=config.hidden_units_per_layer,
        learning_rate=config.learning_rate,
    )


def train_experiment(
    config: ExperimentConfig,
    metadata: dict[str, object],
    teacher_forcing_dir: str | Path,
    model_dir: str | Path,
    history_dir: str | Path,
    epochs: int,
    batch_size: int,
    limit_train_samples: int | None = None,
    limit_val_samples: int | None = None,
) -> dict[str, object]:
    features = np.load(str(metadata["feature_file"])).astype("float32", copy=False)
    train_x, train_y = load_split_arrays(
        teacher_forcing_dir,
        "train",
        features,
        limit_samples=limit_train_samples,
    )
    val_x, val_y = load_split_arrays(
        teacher_forcing_dir,
        "val",
        features,
        limit_samples=limit_val_samples,
    )

    model = build_model(config, metadata)
    started_at = time.perf_counter()
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size,
    )
    elapsed_seconds = time.perf_counter() - started_at

    model_path = Path(model_dir) / f"{config.experiment_id}.weights.h5"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(model_path)

    history_payload = {
        "config": asdict(config),
        "epochs": epochs,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
        "weights_file": str(model_path),
    }
    history_path = Path(history_dir) / f"{config.experiment_id}_history.json"
    save_json(history_payload, history_path)

    return {
        "experiment_id": config.experiment_id,
        "model_kind": config.model_kind,
        "recurrent_layers": config.recurrent_layers,
        "hidden_units": config.hidden_units,
        "epochs": epochs,
        "batch_size": batch_size,
        "elapsed_seconds": elapsed_seconds,
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]),
        "weights_file": str(model_path),
        "history_file": str(history_path),
    }


def select_configs(
    configs: list[ExperimentConfig],
    model_kind: str,
    experiment_id: str | None,
) -> list[ExperimentConfig]:
    selected = configs
    if model_kind != "both":
        selected = [config for config in selected if config.model_kind == model_kind]
    if experiment_id is not None:
        selected = [config for config in selected if config.experiment_id == experiment_id]
    if not selected:
        raise ValueError("No experiment configs matched the requested filters")
    return selected


def main() -> None:
    configs = select_configs(
        default_experiment_configs(),
        model_kind=MODEL_KIND,
        experiment_id=EXPERIMENT_ID,
    )
    save_config_table(configs, DEFAULT_EXPERIMENT_DIR / "decoder_experiment_configs.csv")
    save_json(
        [asdict(config) for config in configs],
        DEFAULT_EXPERIMENT_DIR / "decoder_experiment_configs.json",
    )

    if WRITE_CONFIGS_ONLY:
        print(f"Saved {len(configs)} experiment configs: {DEFAULT_EXPERIMENT_DIR}")
        return

    metadata = load_training_metadata(DEFAULT_PREPROCESSED_DIR, DEFAULT_TEACHER_FORCING_DIR)
    summaries = []
    for config in configs:
        print(f"Training {config.experiment_id}")
        summaries.append(
            train_experiment(
                config=config,
                metadata=metadata,
                teacher_forcing_dir=DEFAULT_TEACHER_FORCING_DIR,
                model_dir=DEFAULT_MODEL_DIR,
                history_dir=DEFAULT_HISTORY_DIR,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                limit_train_samples=LIMIT_TRAIN_SAMPLES,
                limit_val_samples=LIMIT_VAL_SAMPLES,
            )
        )

    save_json(summaries, DEFAULT_EXPERIMENT_DIR / "decoder_training_summary.json")
    print(
        "Saved training summary: "
        f"{DEFAULT_EXPERIMENT_DIR / 'decoder_training_summary.json'}"
    )


if __name__ == "__main__":
    main()
