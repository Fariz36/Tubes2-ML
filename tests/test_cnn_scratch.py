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

from cnn.nn import KerasLocallyConnected2D
from cnn.training import keras_to_scratch

class ScratchCNNParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = np.random.default_rng(1234)

    def test_shared_cnn_matches_keras(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(8, 8, 3)),
                tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(5, activation="softmax"),
            ]
        )

        inputs = self.rng.normal(size=(3, 8, 8, 3)).astype(np.float32)
        expected = model(inputs, training=False).numpy()
        scratch_model = keras_to_scratch(model)
        actual = scratch_model.predict(inputs)

        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_non_shared_cnn_matches_keras(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(6, 6, 2)),
                KerasLocallyConnected2D(3, 3, padding="valid", activation="relu"),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(4, activation="softmax"),
            ]
        )

        inputs = self.rng.normal(size=(2, 6, 6, 2)).astype(np.float32)
        expected = model(inputs, training=False).numpy()
        scratch_model = keras_to_scratch(model)
        actual = scratch_model.predict(inputs)

        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_batch_inference_matches_full_batch(self):
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(8, 8, 3)),
                tf.keras.layers.Conv2D(4, 3, padding="same", activation="relu"),
                tf.keras.layers.AveragePooling2D(pool_size=2),
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(5, activation="softmax"),
            ]
        )

        inputs = self.rng.normal(size=(7, 8, 8, 3)).astype(np.float32)
        scratch_model = keras_to_scratch(model)
        full_batch = scratch_model.predict(inputs)
        chunked_batch = scratch_model.predict(inputs, batch_size=3)

        np.testing.assert_allclose(chunked_batch, full_batch, atol=1e-5, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()