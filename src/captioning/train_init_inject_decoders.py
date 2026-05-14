from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from captioning.decoder import build_init_inject_lstm_decoder, build_init_inject_simple_rnn_decoder
from captioning.train_decoders import (
    DEFAULT_EXPERIMENT_DIR,
    DEFAULT_HISTORY_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_PREPROCESSED_DIR,
    DEFAULT_TEACHER_FORCING_DIR,
    load_split_arrays,
    load_training_metadata,
    save_json,
)


MODEL_KIND = "both"  # "both", "rnn", or "lstm"
EXPERIMENT_ID: str | None = None
EPOCHS = 20
BATCH_SIZE = 64
LIMIT_TRAIN_SAMPLES: int | None = None
LIMIT_VAL_SAMPLES: int | None = None
WRITE_CONFIGS_ONLY = False


@dataclass(frozen=True)
class InitInjectExperimentConfig:
    experiment_id: str
    model_kind: str
    recurrent_layers: int
    hidden_units: int
    embed_dim: int = 256
    learning_rate: float = 1e-3
    injection_method: str = "init-inject"

    @property
    def hidden_units_per_layer(self) -> tuple[int, ...]:
        return (self.hidden_units,) * self.recurrent_layers


def default_experiment_configs() -> list[InitInjectExperimentConfig]:
    configs: list[InitInjectExperimentConfig] = []
    for model_kind in ("rnn", "lstm"):
        for recurrent_layers in (1, 2, 3):
            for hidden_units in (128, 512):
                configs.append(
                    InitInjectExperimentConfig(
                        experiment_id=f"initinject_{model_kind}_layers{recurrent_layers}_hidden{hidden_units}",
                        model_kind=model_kind,
                        recurrent_layers=recurrent_layers,
                        hidden_units=hidden_units,
                    )
                )
    return configs


def save_config_table(configs: list[InitInjectExperimentConfig], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(configs[0]))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for config in configs:
            writer.writerow(asdict(config))


def build_model(config: InitInjectExperimentConfig, metadata: dict[str, object]) -> tf.keras.Model:
    builder = build_init_inject_simple_rnn_decoder if config.model_kind == "rnn" else build_init_inject_lstm_decoder
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
    config: InitInjectExperimentConfig,
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
    train_x, train_y = load_split_arrays(teacher_forcing_dir, "train", features, limit_samples=limit_train_samples)
    val_x, val_y = load_split_arrays(teacher_forcing_dir, "val", features, limit_samples=limit_val_samples)

    model = build_model(config, metadata)
    started_at = time.perf_counter()
    history = model.fit(
        train_x,
        train_y,
        validation_data=(val_x, val_y),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
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
        "injection_method": config.injection_method,
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
    configs: list[InitInjectExperimentConfig],
    model_kind: str,
    experiment_id: str | None,
) -> list[InitInjectExperimentConfig]:
    selected = configs
    if model_kind != "both":
        selected = [config for config in selected if config.model_kind == model_kind]
    if experiment_id is not None:
        selected = [config for config in selected if config.experiment_id == experiment_id]
    if not selected:
        raise ValueError("No init-inject experiment configs matched the requested filters")
    return selected


def main() -> None:
    configs = select_configs(default_experiment_configs(), model_kind=MODEL_KIND, experiment_id=EXPERIMENT_ID)
    save_config_table(configs, DEFAULT_EXPERIMENT_DIR / "initinject_decoder_experiment_configs.csv")
    save_json([asdict(config) for config in configs], DEFAULT_EXPERIMENT_DIR / "initinject_decoder_experiment_configs.json")

    if WRITE_CONFIGS_ONLY:
        print(f"Saved {len(configs)} init-inject experiment configs: {DEFAULT_EXPERIMENT_DIR}")
        return

    metadata = load_training_metadata(DEFAULT_PREPROCESSED_DIR, DEFAULT_TEACHER_FORCING_DIR)
    summaries = []
    for config in tqdm(configs, desc="Training init-inject decoder experiments", unit="experiment"):
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

    summary_path = DEFAULT_EXPERIMENT_DIR / "initinject_decoder_training_summary.json"
    save_json(summaries, summary_path)
    print(f"Saved init-inject training summary: {summary_path}")


if __name__ == "__main__":
    main()
