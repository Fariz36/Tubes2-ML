from math import ceil
import numpy as np

def normalize_tuple(value, length, name):
    if isinstance(value, int):
        values = (value,) * length
    else:
        values = tuple(int(item) for item in value)

    if len(values) != length:
        raise ValueError(f"{name} must have length {length}, got {values}")
    
    if any(item <= 0 for item in values):
        raise ValueError(f"{name} must contain positive integers, got {values}")
    
    return values

def effective_kernel_size(kernel_size, dilation): return kernel_size + (kernel_size - 1) * (dilation - 1)
def compute_conv_output_length(input_length, kernel_size, stride, padding, dilation=1):
    effective_kernel = effective_kernel_size(kernel_size, dilation)
    normalized_padding = padding.lower()
    if normalized_padding == "same":
        return ceil(input_length / stride)
    if normalized_padding == "valid":
        return (input_length - effective_kernel) // stride + 1
    raise ValueError(f"Unsupported padding mode: {padding}")

def compute_padding(input_length, kernel_size, stride, padding, dilation=1):
    normalized_padding = padding.lower()
    effective_kernel = effective_kernel_size(kernel_size, dilation)

    if normalized_padding == "valid":
        return 0, 0
    if normalized_padding != "same":
        raise ValueError(f"Unsupported padding mode: {padding}")

    output_length = ceil(input_length / stride)
    total_padding = max((output_length - 1) * stride + effective_kernel - input_length, 0)
    padding_before = total_padding // 2
    padding_after = total_padding - padding_before
    return padding_before, padding_after

def ensure_4d_batch(inputs):
    array = np.asarray(inputs)
    squeeze_batch = False

    if array.ndim == 3:
        array = array[np.newaxis, ...]
        squeeze_batch = True

    if array.ndim != 4:
        raise ValueError(
            "Expected image tensor with shape (batch, height, width, channels) "
            f"or (height, width, channels), got {array.shape}"
        )

    return array, squeeze_batch

def restore_batch_dim(outputs, squeeze_batch):
    if squeeze_batch:
        return outputs[0]
    return outputs

def extract_image_patches(inputs, kernel_size, strides, padding, dilation_rate=(1, 1)):
    array, squeeze_batch = ensure_4d_batch(inputs)
    kernel_height, kernel_width = kernel_size
    stride_height, stride_width = strides
    dilation_height, dilation_width = dilation_rate

    height = array.shape[1]
    width = array.shape[2]
    pad_top, pad_bottom = compute_padding(
        input_length=height,
        kernel_size=kernel_height,
        stride=stride_height,
        padding=padding,
        dilation=dilation_height,
    )
    pad_left, pad_right = compute_padding(
        input_length=width,
        kernel_size=kernel_width,
        stride=stride_width,
        padding=padding,
        dilation=dilation_width,
    )

    padded = np.pad(
        array,
        pad_width=((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode="constant",
        constant_values=0.0,
    )

    effective_kernel_height = effective_kernel_size(kernel_height, dilation_height)
    effective_kernel_width = effective_kernel_size(kernel_width, dilation_width)

    output_height = compute_conv_output_length(
        input_length=height,
        kernel_size=kernel_height,
        stride=stride_height,
        padding=padding,
        dilation=dilation_height,
    )
    output_width = compute_conv_output_length(
        input_length=width,
        kernel_size=kernel_width,
        stride=stride_width,
        padding=padding,
        dilation=dilation_width,
    )

    if output_height <= 0 or output_width <= 0:
        raise ValueError(
            "Invalid output size. Check kernel_size, strides, padding, and input shape."
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        window_shape=(effective_kernel_height, effective_kernel_width),
        axis=(1, 2),
    )
    windows = windows[:, ::stride_height, ::stride_width, :, :, :]
    windows = windows[:, :output_height, :output_width, :, :, :]
    patches = np.moveaxis(windows, 3, -1)
    patches = patches[:, :, :, ::dilation_height, ::dilation_width, :]
    return patches, squeeze_batch