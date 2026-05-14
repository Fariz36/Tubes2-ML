import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..data import build_tf_dataset, iter_image_batches
from ..nn import macro_f1_score
from .serialization import keras_to_scratch
from .train import build_cnn_classifier, compile_cnn_classifier, evaluate_cnn_classifier, save_history

@dataclass(slots=True)
class CNNExperimentConfig:
    name: str
    conv_filters: tuple
    kernel_sizes: tuple
    pooling_type: str
    dense_units: tuple = (128,)
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 1e-3
    input_shape: tuple = (150, 150, 3)
    use_global_pooling: bool = False
    conv_padding: str = "same"
    activation: str = "relu"
    local_connectivity: bool = False

def _normalize_per_layer(values, length, name):
    if isinstance(values, int):
        normalized = [values] * length
    elif isinstance(values, tuple) and values and isinstance(values[0], int):
        normalized = [tuple(int(item) for item in values)] * length
    else:
        normalized = list(values)

    if len(normalized) != length:
        raise ValueError(f"{name} must have length {length}, got {len(normalized)}")
    return tuple(normalized)

def build_experiment_grid(
    layer_counts,
    filter_variants,
    kernel_variants,
    pooling_types,
    *,
    dense_units=(128,),
    batch_size=32,
    epochs=15,
    learning_rate=1e-3,
    input_shape=(150, 150, 3),
    use_global_pooling=False,
):
    experiments = []

    for layer_count in layer_counts:
        for filter_variant in filter_variants:
            conv_filters = tuple(int(value) for value in _normalize_per_layer(filter_variant, layer_count, "filters"))
            for kernel_variant in kernel_variants:
                kernel_sizes = tuple(
                    tuple(int(item) for item in kernel_size)
                    for kernel_size in _normalize_per_layer(kernel_variant, layer_count, "kernel_sizes")
                )
                for pooling_type in pooling_types:
                    name = (
                        f"cnn_l{layer_count}"
                        f"_f{'-'.join(str(value) for value in conv_filters)}"
                        f"_k{'-'.join(f'{kernel[0]}x{kernel[1]}' for kernel in kernel_sizes)}"
                        f"_{pooling_type}"
                    )
                    experiments.append(
                        CNNExperimentConfig(
                            name=name,
                            conv_filters=conv_filters,
                            kernel_sizes=kernel_sizes,
                            pooling_type=str(pooling_type),
                            dense_units=tuple(int(value) for value in dense_units),
                            batch_size=batch_size,
                            epochs=epochs,
                            learning_rate=learning_rate,
                            input_shape=tuple(int(value) for value in input_shape),
                            use_global_pooling=use_global_pooling,
                        )
                    )

    return experiments

def create_default_callbacks(output_dir, model_name, monitor="val_macro_f1"):
    output_path = Path(output_dir)
    weights_dir = output_path / "weights"
    logs_dir = output_path / "logs"
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = weights_dir / f"{model_name}.weights.h5"
    csv_log_path = logs_dir / f"{model_name}.csv"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=monitor,
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(csv_log_path),
    ]

def train_experiment(
    config: CNNExperimentConfig,
    train_split,
    val_split,
    *,
    num_classes,
    output_dir,
    preprocess_fn=None,
    normalize=True,
):
    model = build_cnn_classifier(
        input_shape=config.input_shape,
        num_classes=num_classes,
        conv_filters=config.conv_filters,
        kernel_sizes=config.kernel_sizes,
        pooling_type=config.pooling_type,
        dense_units=config.dense_units,
        use_global_pooling=config.use_global_pooling,
        conv_padding=config.conv_padding,
        activation=config.activation,
        local_connectivity=config.local_connectivity,
        name=config.name,
    )
    compile_cnn_classifier(
        model=model,
        num_classes=num_classes,
        learning_rate=config.learning_rate,
    )

    target_size = config.input_shape[:2]
    train_dataset = build_tf_dataset(
        train_split,
        target_size=target_size,
        batch_size=config.batch_size,
        shuffle=True,
        normalize=normalize,
        preprocess_fn=preprocess_fn,
    )
    val_dataset = build_tf_dataset(
        val_split,
        target_size=target_size,
        batch_size=config.batch_size,
        shuffle=False,
        normalize=normalize,
        preprocess_fn=preprocess_fn,
    )

    callbacks = create_default_callbacks(output_dir=output_dir, model_name=config.name)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_history(history, output_path / f"{config.name}_history.json")
    model.save(output_path / f"{config.name}_final.keras")
    model.save_weights(output_path / f"{config.name}.weights.h5")

    metrics = evaluate_cnn_classifier(model, val_dataset, verbose=0)
    summary = {
        "config": asdict(config),
        "validation_metrics": metrics,
        "parameter_count": int(model.count_params()),
        "best_epoch": int(np.argmax(history.history["val_macro_f1"]) + 1),
        "best_val_macro_f1": float(np.max(history.history["val_macro_f1"])),
    }
    (output_path / f"{config.name}_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return model, history, summary

def summarize_experiments(summary_paths):
    summaries = []
    for summary_path in summary_paths:
        path = Path(summary_path)
        if not path.exists():
            continue
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    return summaries

def predict_scratch_split(
    scratch_model,
    split,
    *,
    target_size,
    batch_size=32,
    normalize=True,
    preprocess_fn=None,
):
    predictions = []
    for batch in iter_image_batches(
        image_paths=split.image_paths,
        target_size=target_size,
        batch_size=batch_size,
        normalize=normalize,
    ):
        if preprocess_fn is not None:
            batch = preprocess_fn(batch)
        predictions.append(np.asarray(scratch_model.predict(batch, batch_size=batch_size)))

    if not predictions:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(predictions, axis=0)

def compare_keras_and_scratch(
    keras_model,
    split,
    *,
    target_size,
    batch_size=32,
    normalize=True,
    preprocess_fn=None,
):
    keras_dataset = build_tf_dataset(
        split,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False,
        normalize=normalize,
        preprocess_fn=preprocess_fn,
    )
    keras_predictions = keras_model.predict(keras_dataset, verbose=0)
    scratch_model = keras_to_scratch(keras_model)
    scratch_predictions = predict_scratch_split(
        scratch_model,
        split,
        target_size=target_size,
        batch_size=batch_size,
        normalize=normalize,
        preprocess_fn=preprocess_fn,
    )
    return {
        "keras_macro_f1": macro_f1_score(split.labels, keras_predictions, num_classes=keras_predictions.shape[-1]),
        "scratch_macro_f1": macro_f1_score(split.labels, scratch_predictions, num_classes=scratch_predictions.shape[-1]),
        "max_probability_gap": float(np.max(np.abs(keras_predictions - scratch_predictions))),
        "mean_probability_gap": float(np.mean(np.abs(keras_predictions - scratch_predictions))),
    }

def compare_shared_and_non_shared(
    shared_model,
    non_shared_model,
    split,
    *,
    target_size,
    batch_size=32,
    normalize=True,
):
    shared_predictions = shared_model.predict(
        build_tf_dataset(
            split,
            target_size=target_size,
            batch_size=batch_size,
            shuffle=False,
            normalize=normalize,
        ),
        verbose=0,
    )
    non_shared_predictions = non_shared_model.predict(
        build_tf_dataset(
            split,
            target_size=target_size,
            batch_size=batch_size,
            shuffle=False,
            normalize=normalize,
        ),
        verbose=0,
    )
    return {
        "shared_macro_f1": macro_f1_score(split.labels, shared_predictions, num_classes=shared_predictions.shape[-1]),
        "non_shared_macro_f1": macro_f1_score(split.labels, non_shared_predictions, num_classes=non_shared_predictions.shape[-1]),
        "shared_params": int(shared_model.count_params()),
        "non_shared_params": int(non_shared_model.count_params()),
    }