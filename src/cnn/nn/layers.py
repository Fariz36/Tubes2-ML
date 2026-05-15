import numpy as np

from ..core.base import Layer, WeightedLayer
from ..core.ops import (
    ensure_4d_batch,
    extract_image_patches,
    normalize_tuple,
    restore_batch_dim,
)
from .activations import get_activation

class ActivationLayer(Layer):
    def __init__(self, activation, name=None):
        super().__init__(name=name)
        self.activation = activation or "linear"
        self.activation_fn = get_activation(self.activation)

    def forward(self, inputs):
        return self.activation_fn(np.asarray(inputs))

    def get_config(self):
        config = super().get_config()
        config.update({"activation": self.activation})
        return config

class Conv2D(WeightedLayer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        activation=None,
        use_bias=True,
        dilation_rate=(1, 1),
        name=None,
    ):
        super().__init__(name=name)
        self.filters = int(filters)
        self.kernel_size = normalize_tuple(kernel_size, 2, "kernel_size")
        self.strides = normalize_tuple(strides, 2, "strides")
        self.padding = padding.lower()
        self.activation = activation or "linear"
        self.activation_fn = get_activation(self.activation)
        self.use_bias = use_bias
        self.dilation_rate = normalize_tuple(dilation_rate, 2, "dilation_rate")
        self.kernel = None
        self.bias = None

    def set_weights(self, weights):
        if len(weights) not in {1, 2}:
            raise ValueError(f"Conv2D expects 1 or 2 tensors, got {len(weights)}")

        kernel = np.asarray(weights[0], dtype=np.float32)
        if kernel.ndim != 4:
            raise ValueError(f"Conv2D kernel must be 4D, got {kernel.shape}")
        if kernel.shape[:2] != self.kernel_size:
            raise ValueError(
                f"Conv2D kernel size mismatch, expected {self.kernel_size}, got {kernel.shape[:2]}"
            )
        if kernel.shape[-1] != self.filters:
            raise ValueError(
                f"Conv2D filter count mismatch, expected {self.filters}, got {kernel.shape[-1]}"
            )

        self.kernel = kernel
        self.bias = None
        if len(weights) == 2:
            bias = np.asarray(weights[1], dtype=np.float32)
            if bias.shape != (self.filters,):
                raise ValueError(
                    f"Conv2D bias must have shape {(self.filters,)}, got {bias.shape}"
                )
            self.bias = bias
        elif self.use_bias:
            self.bias = np.zeros((self.filters,), dtype=np.float32)

    def get_weights(self):
        if self.kernel is None:
            return []
        if self.bias is None:
            return [self.kernel.copy()]
        return [self.kernel.copy(), self.bias.copy()]

    def forward(self, inputs):
        if self.kernel is None:
            raise RuntimeError("Conv2D weights are not initialized")

        patches, squeeze_batch = extract_image_patches(
            inputs=inputs,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
        )
        outputs = np.einsum("nhwklc,klcf->nhwf", patches, self.kernel, optimize=True)
        if self.bias is not None:
            outputs = outputs + self.bias.reshape((1, 1, 1, -1))
        outputs = self.activation_fn(outputs)
        return restore_batch_dim(outputs, squeeze_batch)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": self.activation,
                "use_bias": self.use_bias,
                "dilation_rate": self.dilation_rate,
            }
        )
        return config

class LocallyConnected2D(WeightedLayer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        activation=None,
        use_bias=True,
        dilation_rate=(1, 1),
        name=None,
    ):
        super().__init__(name=name)
        self.filters = int(filters)
        self.kernel_size = normalize_tuple(kernel_size, 2, "kernel_size")
        self.strides = normalize_tuple(strides, 2, "strides")
        self.padding = padding.lower()
        self.activation = activation or "linear"
        self.activation_fn = get_activation(self.activation)
        self.use_bias = use_bias
        self.dilation_rate = normalize_tuple(dilation_rate, 2, "dilation_rate")
        self.kernel = None
        self.bias = None

    def set_weights(self, weights):
        if len(weights) not in {1, 2}:
            raise ValueError(
                f"LocallyConnected2D expects 1 or 2 tensors, got {len(weights)}"
            )

        kernel = np.asarray(weights[0], dtype=np.float32)
        if kernel.ndim != 3:
            raise ValueError(f"LocallyConnected2D kernel must be 3D, got {kernel.shape}")
        if kernel.shape[-1] != self.filters:
            raise ValueError(
                "LocallyConnected2D filter count mismatch, "
                f"expected {self.filters}, got {kernel.shape[-1]}"
            )

        self.kernel = kernel
        self.bias = None
        if len(weights) == 2:
            bias = np.asarray(weights[1], dtype=np.float32)
            if bias.ndim not in {1, 2}:
                raise ValueError(
                    "LocallyConnected2D bias must be 1D or 2D, "
                    f"got shape {bias.shape}"
                )
            self.bias = bias
        elif self.use_bias:
            self.bias = np.zeros((self.filters,), dtype=np.float32)

    def get_weights(self):
        if self.kernel is None:
            return []
        if self.bias is None:
            return [self.kernel.copy()]
        return [self.kernel.copy(), self.bias.copy()]

    def forward(self, inputs):
        if self.kernel is None:
            raise RuntimeError("LocallyConnected2D weights are not initialized")

        patches, squeeze_batch = extract_image_patches(
            inputs=inputs,
            kernel_size=self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
        )
        batch_size, output_height, output_width = patches.shape[:3]
        flattened_patches = patches.reshape(batch_size, output_height * output_width, -1)

        expected_shape = (
            output_height * output_width,
            flattened_patches.shape[-1],
            self.filters,
        )
        if self.kernel.shape != expected_shape:
            raise ValueError(
                "LocallyConnected2D kernel shape mismatch, "
                f"expected {expected_shape}, got {self.kernel.shape}"
            )

        outputs = np.einsum("npc,pcf->npf", flattened_patches, self.kernel, optimize=True)
        if self.bias is not None:
            if self.bias.ndim == 1:
                outputs = outputs + self.bias.reshape((1, 1, -1))
            else:
                outputs = outputs + self.bias.reshape((1, output_height * output_width, -1))

        outputs = outputs.reshape(batch_size, output_height, output_width, self.filters)
        outputs = self.activation_fn(outputs)
        return restore_batch_dim(outputs, squeeze_batch)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": self.activation,
                "use_bias": self.use_bias,
                "dilation_rate": self.dilation_rate,
            }
        )
        return config

class _Pooling2D(Layer):
    def __init__(
        self,
        pool_size=(2, 2),
        strides=None,
        padding="valid",
        name=None,
    ):
        super().__init__(name=name)
        self.pool_size = normalize_tuple(pool_size, 2, "pool_size")
        self.strides = normalize_tuple(strides or pool_size, 2, "strides")
        self.padding = padding.lower()

    def _pool(self, patches):
        raise NotImplementedError

    def forward(self, inputs):
        patches, squeeze_batch = extract_image_patches(
            inputs=inputs,
            kernel_size=self.pool_size,
            strides=self.strides,
            padding=self.padding,
        )
        outputs = self._pool(patches)
        return restore_batch_dim(outputs, squeeze_batch)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "pool_size": self.pool_size,
                "strides": self.strides,
                "padding": self.padding,
            }
        )
        return config

class MaxPooling2D(_Pooling2D):
    def _pool(self, patches):
        return np.max(patches, axis=(3, 4))

class AveragePooling2D(_Pooling2D):
    def _pool(self, patches):
        return np.mean(patches, axis=(3, 4))

class GlobalAveragePooling2D(Layer):
    def forward(self, inputs):
        array, squeeze_batch = ensure_4d_batch(inputs)
        outputs = np.mean(array, axis=(1, 2))
        return restore_batch_dim(outputs, squeeze_batch)

class GlobalMaxPooling2D(Layer):
    def forward(self, inputs):
        array, squeeze_batch = ensure_4d_batch(inputs)
        outputs = np.max(array, axis=(1, 2))
        return restore_batch_dim(outputs, squeeze_batch)

class Flatten(Layer):
    def forward(self, inputs):
        array = np.asarray(inputs)
        if array.ndim == 1:
            return array
        if array.ndim == 2:
            return array
        if array.ndim == 3:
            return array.reshape(-1, order="C")
        if array.ndim < 2:
            raise ValueError(f"Flatten expects rank >= 2, got {array.shape}")
        return array.reshape(array.shape[0], -1, order="C")

class Dense(WeightedLayer):
    def __init__(
        self,
        units,
        activation=None,
        use_bias=True,
        name=None,
    ):
        super().__init__(name=name)
        self.units = int(units)
        self.activation = activation or "linear"
        self.activation_fn = get_activation(self.activation)
        self.use_bias = use_bias
        self.kernel = None
        self.bias = None

    def set_weights(self, weights):
        if len(weights) not in {1, 2}:
            raise ValueError(f"Dense expects 1 or 2 tensors, got {len(weights)}")

        kernel = np.asarray(weights[0], dtype=np.float32)
        if kernel.ndim != 2:
            raise ValueError(f"Dense kernel must be 2D, got {kernel.shape}")
        if kernel.shape[-1] != self.units:
            raise ValueError(f"Dense units mismatch, expected {self.units}, got {kernel.shape[-1]}")

        self.kernel = kernel
        self.bias = None
        if len(weights) == 2:
            bias = np.asarray(weights[1], dtype=np.float32)
            if bias.shape != (self.units,):
                raise ValueError(f"Dense bias must have shape {(self.units,)}, got {bias.shape}")
            self.bias = bias
        elif self.use_bias:
            self.bias = np.zeros((self.units,), dtype=np.float32)

    def get_weights(self):
        if self.kernel is None:
            return []
        if self.bias is None:
            return [self.kernel.copy()]
        return [self.kernel.copy(), self.bias.copy()]

    def forward(self, inputs):
        if self.kernel is None:
            raise RuntimeError("Dense weights are not initialized")

        array = np.asarray(inputs, dtype=np.float32)
        if array.shape[-1] != self.kernel.shape[0]:
            raise ValueError(
                "Dense input feature mismatch, "
                f"expected {self.kernel.shape[0]}, got {array.shape[-1]}"
            )

        outputs = np.tensordot(array, self.kernel, axes=([-1], [0]))
        if self.bias is not None:
            outputs = outputs + self.bias
        return self.activation_fn(outputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": self.activation,
                "use_bias": self.use_bias,
            }
        )
        return config