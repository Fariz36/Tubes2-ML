import json
from pathlib import Path

import numpy as np

from ..core.model import Sequential
from ..nn.keras_layers import KerasLocallyConnected2D
from ..nn.metrics import SparseMacroF1
from ..nn.layers import (
    ActivationLayer,
    AveragePooling2D,
    Conv2D,
    Dense,
    Flatten,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    LocallyConnected2D,
    MaxPooling2D,
)
from .train import build_cnn_classifier, compile_cnn_classifier

def load_keras_model(model_or_path, compile=True):
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow is required to load a Keras model. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    if hasattr(model_or_path, "layers"):
        return model_or_path

    path = Path(model_or_path)
    if not path.exists():
        raise FileNotFoundError(f"Keras model not found: {path}")

    return tf.keras.models.load_model(
        path,
        custom_objects={
            "KerasLocallyConnected2D": KerasLocallyConnected2D,
            "SparseMacroF1": SparseMacroF1,
        },
        compile=compile,
    )

def load_saved_classifier(model_dir, model_name, compile=False):
    directory = Path(model_dir)
    final_model_path = directory / f"{model_name}_final.keras"
    if final_model_path.exists():
        return load_keras_model(final_model_path, compile=compile)

    summary_path = directory / f"{model_name}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Model summary not found: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = summary["config"]
    keras_model = build_cnn_classifier(
        input_shape=tuple(config["input_shape"]),
        num_classes=int(summary["num_classes"]),
        conv_filters=tuple(config["conv_filters"]),
        kernel_sizes=tuple(tuple(kernel) for kernel in config["kernel_sizes"]),
        pooling_type=str(config["pooling_type"]),
        dense_units=tuple(config["dense_units"]),
        activation=str(config["activation"]),
        conv_padding=str(config["conv_padding"]),
        use_global_pooling=bool(config["use_global_pooling"]),
        local_connectivity=bool(config.get("local_connectivity", False)),
        name=str(config["name"]),
    )

    weights_h5_path = directory / f"{model_name}.weights.h5"
    weights_json_path = directory / f"{model_name}.weights.json"
    if weights_h5_path.exists():
        keras_model.load_weights(weights_h5_path)
    elif weights_json_path.exists():
        keras_model.load_weights(weights_json_path)
    else:
        raise FileNotFoundError(
            f"No supported weight artifact found for {model_name} in {directory}"
        )

    if compile:
        compile_cnn_classifier(
            model=keras_model,
            num_classes=int(keras_model.output_shape[-1]),
            learning_rate=float(config["learning_rate"]),
        )
    return keras_model

def _serialize_activation(activation):
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow is required to inspect Keras activations. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    serialized = tf.keras.activations.serialize(activation)
    if isinstance(serialized, dict):
        return str(serialized.get("class_name", "linear")).lower()
    return str(serialized).lower()

def keras_to_scratch(model_or_path, name=None):
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow is required to convert a Keras model. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error

    keras_model = load_keras_model(model_or_path, compile=False)
    scratch_layers = []

    for layer in keras_model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue

        if isinstance(layer, tf.keras.layers.Conv2D):
            scratch_layer = Conv2D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                activation=_serialize_activation(layer.activation),
                use_bias=layer.use_bias,
                dilation_rate=layer.dilation_rate,
                name=layer.name,
            )
            scratch_layer.set_weights([np.asarray(weight) for weight in layer.get_weights()])
        elif isinstance(layer, KerasLocallyConnected2D):
            scratch_layer = LocallyConnected2D(
                filters=layer.filters,
                kernel_size=layer.kernel_size,
                strides=layer.strides,
                padding=layer.padding,
                activation=layer.activation_name,
                use_bias=layer.use_bias,
                dilation_rate=layer.dilation_rate,
                name=layer.name,
            )
            scratch_layer.set_weights([np.asarray(weight) for weight in layer.get_weights()])
        elif isinstance(layer, tf.keras.layers.MaxPooling2D):
            scratch_layer = MaxPooling2D(
                pool_size=layer.pool_size,
                strides=layer.strides,
                padding=layer.padding,
                name=layer.name,
            )
        elif isinstance(layer, tf.keras.layers.AveragePooling2D):
            scratch_layer = AveragePooling2D(
                pool_size=layer.pool_size,
                strides=layer.strides,
                padding=layer.padding,
                name=layer.name,
            )
        elif isinstance(layer, tf.keras.layers.GlobalAveragePooling2D):
            scratch_layer = GlobalAveragePooling2D(name=layer.name)
        elif isinstance(layer, tf.keras.layers.GlobalMaxPooling2D):
            scratch_layer = GlobalMaxPooling2D(name=layer.name)
        elif isinstance(layer, tf.keras.layers.Flatten):
            scratch_layer = Flatten(name=layer.name)
        elif isinstance(layer, tf.keras.layers.Dense):
            scratch_layer = Dense(
                units=layer.units,
                activation=_serialize_activation(layer.activation),
                use_bias=layer.use_bias,
                name=layer.name,
            )
            scratch_layer.set_weights([np.asarray(weight) for weight in layer.get_weights()])
        elif isinstance(layer, tf.keras.layers.ReLU):
            scratch_layer = ActivationLayer(activation="relu", name=layer.name)
        elif isinstance(layer, tf.keras.layers.Softmax):
            scratch_layer = ActivationLayer(activation="softmax", name=layer.name)
        elif isinstance(layer, tf.keras.layers.Activation):
            scratch_layer = ActivationLayer(
                activation=_serialize_activation(layer.activation),
                name=layer.name,
            )
        else:
            raise TypeError(
                f"Unsupported Keras layer for scratch conversion: {layer.__class__.__name__}"
            )

        scratch_layers.append(scratch_layer)

    return Sequential(layers=scratch_layers, name=name or keras_model.name)