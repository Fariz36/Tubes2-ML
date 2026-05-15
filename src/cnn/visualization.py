from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .nn.keras_layers import KerasLocallyConnected2D

def _import_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(
            "TensorFlow is required for CNN visualization utilities. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from error
    return tf

def _prepare_image_batch(images):
    batch = np.asarray(images, dtype=np.float32)
    if batch.ndim == 3:
        batch = batch[np.newaxis, ...]
    if batch.ndim != 4:
        raise ValueError(
            "Expected image batch with shape (batch, height, width, channels) "
            f"or single image (height, width, channels), got {batch.shape}"
        )
    return batch

def _call_model_once(model, batch):
    model(batch, training=False)
    return model

def _dense_logits(tf, layer, inputs):
    outputs = tf.tensordot(inputs, layer.kernel, axes=[[-1], [0]])
    if layer.use_bias:
        outputs = tf.nn.bias_add(outputs, layer.bias)
    return outputs

def _forward_to_class_scores(tf, model, start_index, inputs):
    outputs = inputs
    last_index = len(model.layers) - 1

    for index, layer in enumerate(model.layers[start_index:], start=start_index):
        is_last_layer = index == last_index

        if is_last_layer and isinstance(layer, tf.keras.layers.Dense):
            activation = getattr(layer, "activation", None)
            if activation == tf.keras.activations.softmax:
                return _dense_logits(tf, layer, outputs)

        if is_last_layer and isinstance(layer, tf.keras.layers.Softmax):
            return outputs

        if is_last_layer and isinstance(layer, tf.keras.layers.Activation):
            if getattr(layer, "activation", None) == tf.keras.activations.softmax:
                return outputs

        outputs = layer(outputs)

    return outputs

def list_feature_layers(model):
    tf = _import_tensorflow()
    feature_layer_types = (tf.keras.layers.Conv2D, KerasLocallyConnected2D)
    return [layer.name for layer in model.layers if isinstance(layer, feature_layer_types)]

def get_last_feature_layer(model):
    feature_layers = list_feature_layers(model)
    if not feature_layers:
        raise ValueError("Model does not contain a convolutional feature layer.")
    return feature_layers[-1]

def extract_feature_maps(model, images, layer_names=None):
    tf = _import_tensorflow()
    batch = _prepare_image_batch(images)
    _call_model_once(model, batch)

    selected_layers = list(layer_names or list_feature_layers(model))
    if not selected_layers:
        raise ValueError("No feature layers selected for feature map extraction.")

    feature_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output for layer_name in selected_layers],
    )
    outputs = feature_model(batch, training=False)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]
    return {name: np.asarray(output) for name, output in zip(selected_layers, outputs)}

def plot_feature_maps(
    feature_maps,
    layer_name=None,
    image_index=0,
    max_maps=16,
    cols=4,
    normalize_each=True,
):
    if isinstance(feature_maps, dict):
        if layer_name is None:
            layer_name = next(iter(feature_maps))
        maps = np.asarray(feature_maps[layer_name])
    else:
        maps = np.asarray(feature_maps)
        layer_name = layer_name or "feature_maps"

    if maps.ndim != 4:
        raise ValueError(f"Expected feature maps with shape (batch, height, width, channels), got {maps.shape}")
    if image_index >= maps.shape[0]:
        raise IndexError(f"image_index {image_index} out of range for batch size {maps.shape[0]}")

    maps = maps[image_index]
    num_maps = min(max_maps, maps.shape[-1])
    rows = int(np.ceil(num_maps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for index in range(rows * cols):
        axis = axes.flat[index]
        axis.axis("off")
        if index >= num_maps:
            continue

        feature_map = maps[:, :, index]
        if normalize_each:
            feature_min = feature_map.min()
            feature_max = feature_map.max()
            if feature_max > feature_min:
                feature_map = (feature_map - feature_min) / (feature_max - feature_min)
        axis.imshow(feature_map, cmap="viridis")
        axis.set_title(f"{layer_name}[{index}]")

    fig.tight_layout()
    return fig

def make_gradcam_heatmap(model, image, layer_name=None, class_index=None):
    tf = _import_tensorflow()
    batch = _prepare_image_batch(image)
    if batch.shape[0] != 1:
        raise ValueError("Grad-CAM expects a single image. Pass one image, not a batch.")

    _call_model_once(model, batch)
    layer_name = layer_name or get_last_feature_layer(model)
    batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
    target_layer_index = next(
        (index for index, layer in enumerate(model.layers) if layer.name == layer_name),
        None,
    )
    if target_layer_index is None:
        raise ValueError(f"Layer not found in model: {layer_name}")

    conv_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(layer_name).output,
    )

    with tf.GradientTape() as tape:
        conv_output = conv_model(batch_tensor, training=False)
        tape.watch(conv_output)
        scores = _forward_to_class_scores(
            tf,
            model,
            target_layer_index + 1,
            conv_output,
        )
        if class_index is None:
            class_index = int(tf.argmax(scores[0]).numpy())
        score = scores[:, class_index]

    gradients = tape.gradient(score, conv_output)
    pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_gradients[tf.newaxis, tf.newaxis, :], axis=-1)
    heatmap = tf.nn.relu(heatmap)
    max_value = tf.reduce_max(heatmap)
    if float(max_value.numpy()) > 0.0:
        heatmap = heatmap / max_value

    return np.asarray(heatmap), class_index, layer_name

def resize_heatmap(heatmap, target_size):
    tf = _import_tensorflow()
    resized = tf.image.resize(
        np.asarray(heatmap, dtype=np.float32)[..., np.newaxis],
        target_size,
        method="bilinear",
    )
    return np.asarray(resized[..., 0])

def overlay_heatmap(image, heatmap, alpha=0.4, cmap="jet"):
    image_array = np.asarray(image, dtype=np.float32)
    if image_array.ndim != 3:
        raise ValueError(f"Expected image with shape (height, width, channels), got {image_array.shape}")

    if image_array.max(initial=0.0) > 1.0:
        image_array = image_array / 255.0

    resized_heatmap = resize_heatmap(heatmap, image_array.shape[:2])
    colored_heatmap = plt.get_cmap(cmap)(resized_heatmap)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * image_array + alpha * colored_heatmap
    return np.clip(overlay, 0.0, 1.0)

def plot_gradcam(image, heatmap, alpha=0.4, cmap="jet", title=None):
    overlay = overlay_heatmap(image, heatmap, alpha=alpha, cmap=cmap)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(np.asarray(image))
    axes[0].set_title("Image")
    axes[1].imshow(heatmap, cmap=cmap)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title(title or "Overlay")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    return fig

def save_figure(fig, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    return path