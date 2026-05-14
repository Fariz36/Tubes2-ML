import os
import sys
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cnn import (
    extract_feature_maps,
    get_last_feature_layer,
    list_feature_layers,
    make_gradcam_heatmap,
    overlay_heatmap,
)

class CNNVisualizationTests(unittest.TestCase):
    def test_feature_map_extraction_returns_conv_outputs(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(16, 16, 3)),
                tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="conv_1"),
                tf.keras.layers.MaxPooling2D(pool_size=2),
                tf.keras.layers.Conv2D(6, 3, padding="same", activation="relu", name="conv_2"),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )
        inputs = np.random.default_rng(0).random((1, 16, 16, 3), dtype=np.float32)

        layer_names = list_feature_layers(model)
        feature_maps = extract_feature_maps(model, inputs, layer_names=layer_names)

        self.assertEqual(layer_names, ["conv_1", "conv_2"])
        self.assertEqual(set(feature_maps.keys()), {"conv_1", "conv_2"})
        self.assertEqual(feature_maps["conv_1"].shape, (1, 16, 16, 4))
        self.assertEqual(feature_maps["conv_2"].shape, (1, 8, 8, 6))

    def test_gradcam_returns_normalized_heatmap_and_overlay(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(16, 16, 3)),
                tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu", name="conv_1"),
                tf.keras.layers.MaxPooling2D(pool_size=2),
                tf.keras.layers.Conv2D(6, 3, padding="same", activation="relu", name="conv_2"),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )
        image = np.random.default_rng(1).random((16, 16, 3), dtype=np.float32)

        heatmap, class_index, layer_name = make_gradcam_heatmap(
            model,
            image,
            layer_name=get_last_feature_layer(model),
        )
        overlay = overlay_heatmap(image, heatmap)

        self.assertEqual(layer_name, "conv_2")
        self.assertIsInstance(class_index, int)
        self.assertEqual(heatmap.ndim, 2)
        self.assertGreaterEqual(float(heatmap.min(initial=0.0)), 0.0)
        self.assertLessEqual(float(heatmap.max(initial=0.0)), 1.0)
        self.assertEqual(overlay.shape, image.shape)

    def test_gradcam_matches_manual_pre_softmax_computation(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(8, 8, 1)),
                tf.keras.layers.Conv2D(3, 3, padding="same", activation="relu", name="conv_1"),
                tf.keras.layers.GlobalAveragePooling2D(name="global_pool"),
                tf.keras.layers.Dense(2, activation="softmax", name="classifier"),
            ]
        )
        image = np.random.default_rng(2).random((8, 8, 1), dtype=np.float32)
        heatmap, class_index, _ = make_gradcam_heatmap(model, image, layer_name="conv_1")

        image_batch = image[np.newaxis, ...]
        conv_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.get_layer("conv_1").output,
        )
        classifier = model.get_layer("classifier")

        with tf.GradientTape() as tape:
            conv_output = conv_model(image_batch, training=False)
            tape.watch(conv_output)
            pooled = tf.reduce_mean(conv_output, axis=(1, 2))
            scores = tf.tensordot(pooled, classifier.kernel, axes=[[-1], [0]])
            scores = tf.nn.bias_add(scores, classifier.bias)
            target_class = int(tf.argmax(scores[0]).numpy())
            score = scores[:, target_class]

        gradients = tape.gradient(score, conv_output)
        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        manual_heatmap = tf.reduce_sum(
            conv_output[0] * pooled_gradients[tf.newaxis, tf.newaxis, :],
            axis=-1,
        )
        manual_heatmap = tf.nn.relu(manual_heatmap)
        manual_max = tf.reduce_max(manual_heatmap)
        if float(manual_max.numpy()) > 0.0:
            manual_heatmap = manual_heatmap / manual_max

        self.assertEqual(class_index, target_class)
        np.testing.assert_allclose(
            heatmap,
            np.asarray(manual_heatmap),
            rtol=1e-5,
            atol=1e-5,
        )

if __name__ == "__main__":
    unittest.main()