import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf

from ..nn.keras_layers import KerasLocallyConnected2D
from ..nn.metrics import SparseMacroF1

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def _normalize_sequence(values, length, name):
    if isinstance(values, int):
        return [values] * length
    if isinstance(values, tuple) and values and all(isinstance(item, int) for item in values):
        return [values] * length

    normalized = list(values)
    if len(normalized) != length:
        raise ValueError(f"{name} must have length {length}, got {len(normalized)}")
    return normalized

def build_cnn_classifier(
    input_shape,
    num_classes,
    conv_filters,
    kernel_sizes,
    pooling_type="max",
    dense_units=(128,),
    activation="relu",
    classifier_activation="softmax",
    pooling_size=(2, 2),
    pooling_strides=None,
    conv_padding="same",
    use_global_pooling=False,
    local_connectivity=False,
    dropout_rates=None,
    name="cnn_classifier",
):
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than 0")
    if not conv_filters:
        raise ValueError("conv_filters must not be empty")

    kernel_sizes = _normalize_sequence(kernel_sizes, len(conv_filters), "kernel_sizes")
    dropout_rates = list(dropout_rates or [])
    pooling_type_normalized = pooling_type.lower()
    if pooling_type_normalized not in {"max", "avg", "average"}:
        raise ValueError("pooling_type must be 'max' or 'avg'")

    classifier = tf.keras.Sequential(name=name)
    classifier.add(tf.keras.layers.Input(shape=input_shape, name="input"))

    for index, (filters, kernel_size) in enumerate(zip(conv_filters, kernel_sizes, strict=True)):
        layer_name = f"block{index + 1}"
        if local_connectivity:
            classifier.add(
                KerasLocallyConnected2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    activation=activation,
                    name=f"{layer_name}_local",
                )
            )
        else:
            classifier.add(
                tf.keras.layers.Conv2D(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    activation=activation,
                    name=f"{layer_name}_conv",
                )
            )

        if pooling_type_normalized == "max":
            classifier.add(
                tf.keras.layers.MaxPooling2D(
                    pool_size=pooling_size,
                    strides=pooling_strides,
                    name=f"{layer_name}_pool",
                )
            )
        else:
            classifier.add(
                tf.keras.layers.AveragePooling2D(
                    pool_size=pooling_size,
                    strides=pooling_strides,
                    name=f"{layer_name}_pool",
                )
            )

    if use_global_pooling:
        classifier.add(tf.keras.layers.GlobalAveragePooling2D(name="global_pool"))
    else:
        classifier.add(tf.keras.layers.Flatten(name="flatten"))

    for index, units in enumerate(dense_units):
        classifier.add(
            tf.keras.layers.Dense(units, activation=activation, name=f"dense_{index + 1}")
        )
        if index < len(dropout_rates):
            classifier.add(
                tf.keras.layers.Dropout(dropout_rates[index], name=f"dropout_{index + 1}")
            )

    classifier.add(
        tf.keras.layers.Dense(
            num_classes,
            activation=classifier_activation,
            name="classifier",
        )
    )
    return classifier

def compile_cnn_classifier(
    model,
    num_classes,
    learning_rate=1e-3,
):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            SparseMacroF1(num_classes=num_classes, name="macro_f1"),
        ],
    )
    return model

def _format_value(value):
    if isinstance(value, str):
        return f"'{value}'"
    return repr(value)

def _format_activation_name(activation):
    if activation in (None, "linear"):
        return None
    if isinstance(activation, dict):
        return activation.get("config", {}).get("activation")
    return activation

def format_model_architecture(model, variable_name="classifier"):
    lines = [f'{variable_name} = tf.keras.Sequential(name="{model.name}")']
    input_shape = tuple(int(value) for value in model.input_shape[1:])

    for index, layer in enumerate(model.layers):
        config = layer.get_config()
        args = []

        if isinstance(layer, tf.keras.layers.Conv2D):
            args.append(str(layer.filters))
            args.append(repr(tuple(layer.kernel_size)))
            if index == 0:
                args.append(f"input_shape={repr(input_shape)}")
            if layer.padding != "valid":
                args.append(f"padding='{layer.padding}'")
            activation = _format_activation_name(config.get("activation"))
            if activation is not None:
                args.append(f"activation='{activation}'")
            layer_code = f"Conv2D({', '.join(args)})"
        elif isinstance(layer, KerasLocallyConnected2D):
            args.append(str(layer.filters))
            args.append(repr(tuple(layer.kernel_size)))
            if index == 0:
                args.append(f"input_shape={repr(input_shape)}")
            if layer.padding != "valid":
                args.append(f"padding='{layer.padding}'")
            if layer.activation_name not in (None, "linear"):
                args.append(f"activation='{layer.activation_name}'")
            layer_code = f"KerasLocallyConnected2D({', '.join(args)})"
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            args.append(f"pool_size={repr(tuple(layer.pool_size))}")
            if layer.strides is not None:
                args.append(f"strides={repr(tuple(layer.strides))}")
            layer_code = f"MaxPooling2D({', '.join(args)})"
        elif isinstance(layer, tf.keras.layers.AveragePooling2D):
            args.append(f"pool_size={repr(tuple(layer.pool_size))}")
            if layer.strides is not None:
                args.append(f"strides={repr(tuple(layer.strides))}")
            layer_code = f"AveragePooling2D({', '.join(args)})"
        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            layer_code = "GlobalAveragePooling2D()"
        elif isinstance(layer, tf.keras.layers.GlobalMaxPooling2D):
            layer_code = "GlobalMaxPooling2D()"
        elif isinstance(layer, tf.keras.layers.Flatten):
            layer_code = "Flatten()"
        elif isinstance(layer, tf.keras.layers.Dense):
            args.append(str(layer.units))
            activation = _format_activation_name(config.get("activation"))
            if activation is not None:
                args.append(f"activation='{activation}'")
            layer_code = f"Dense({', '.join(args)})"
        elif isinstance(layer, tf.keras.layers.Dropout):
            layer_code = f"Dropout({layer.rate})"
        elif isinstance(layer, tf.keras.layers.ReLU):
            layer_code = "ReLU()"
        elif isinstance(layer, tf.keras.layers.Softmax):
            layer_code = "Softmax()"
        elif isinstance(layer, tf.keras.layers.Activation):
            activation = _format_activation_name(config.get("activation"))
            layer_code = f"Activation('{activation}')"
        else:
            layer_code = f"{layer.__class__.__name__}({', '.join(f'{key}={_format_value(value)}' for key, value in config.items() if key != 'name')})"

        lines.append(f"{variable_name}.add(tf.keras.layers.{layer_code})")

    return "\n".join(lines)

def summarize_model_layers(model):
    summary = []
    for layer in model.layers:
        config = layer.get_config()
        summary.append(
            {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "trainable": bool(layer.trainable),
                "params": int(layer.count_params()),
                "filters": config.get("filters"),
                "kernel_size": config.get("kernel_size"),
                "pool_size": config.get("pool_size"),
                "strides": config.get("strides"),
                "padding": config.get("padding"),
                "units": config.get("units"),
                "activation": config.get("activation"),
            }
        )
    return summary

def evaluate_cnn_classifier(
    model,
    dataset,
    verbose=0,
):
    metric_values = model.evaluate(dataset, return_dict=True, verbose=verbose)
    return {key: float(value) for key, value in metric_values.items()}

def save_history(history, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: [float(value) for value in values] for key, values in history.history.items()}
    path.write_text(json.dumps(serializable, indent=2) + "\n", encoding="utf-8")