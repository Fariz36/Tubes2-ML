import tensorflow as tf

from ..core.ops import compute_conv_output_length, normalize_tuple

@tf.keras.utils.register_keras_serializable(package="cnn")
class KerasLocallyConnected2D(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        activation=None,
        use_bias=True,
        dilation_rate=(1, 1),
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.filters = int(filters)
        self.kernel_size = normalize_tuple(kernel_size, 2, "kernel_size")
        self.strides = normalize_tuple(strides, 2, "strides")
        self.padding = padding.lower()
        self.activation = tf.keras.activations.get(activation)
        self.activation_name = tf.keras.activations.serialize(self.activation)
        self.use_bias = use_bias
        self.dilation_rate = normalize_tuple(dilation_rate, 2, "dilation_rate")
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.output_height = None
        self.output_width = None

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                "KerasLocallyConnected2D expects inputs with rank 4 "
                f"(batch, height, width, channels), got {input_shape}"
            )

        input_height = int(input_shape[1])
        input_width = int(input_shape[2])
        channels = int(input_shape[3])

        self.output_height = compute_conv_output_length(
            input_length=input_height,
            kernel_size=self.kernel_size[0],
            stride=self.strides[0],
            padding=self.padding,
            dilation=self.dilation_rate[0],
        )
        self.output_width = compute_conv_output_length(
            input_length=input_width,
            kernel_size=self.kernel_size[1],
            stride=self.strides[1],
            padding=self.padding,
            dilation=self.dilation_rate[1],
        )
        if self.output_height <= 0 or self.output_width <= 0:
            raise ValueError(
                "Invalid output shape for KerasLocallyConnected2D. "
                "Check kernel_size, strides, padding, dilation_rate, and input shape."
            )

        patch_dim = self.kernel_size[0] * self.kernel_size[1] * channels
        self.kernel = self.add_weight(
            name="kernel",
            shape=(self.output_height * self.output_width, patch_dim, self.filters),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.output_height * self.output_width, self.filters),
                initializer=self.bias_initializer,
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs):
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.kernel_size[0], self.kernel_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, self.dilation_rate[0], self.dilation_rate[1], 1],
            padding=self.padding.upper(),
        )
        batch_size = tf.shape(patches)[0]
        patches = tf.reshape(
            patches,
            [batch_size, self.output_height * self.output_width, tf.shape(patches)[-1]],
        )
        outputs = tf.einsum("bpc,pcf->bpf", patches, self.kernel)
        if self.bias is not None:
            outputs = outputs + self.bias[tf.newaxis, ...]
        outputs = tf.reshape(outputs, [batch_size, self.output_height, self.output_width, self.filters])
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        return tf.TensorShape((input_shape[0], self.output_height, self.output_width, self.filters))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "activation": self.activation_name,
                "use_bias": self.use_bias,
                "dilation_rate": self.dilation_rate,
                "kernel_initializer": tf.keras.initializers.serialize(self.kernel_initializer),
                "bias_initializer": tf.keras.initializers.serialize(self.bias_initializer),
            }
        )
        return config