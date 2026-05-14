import numpy as np

def _to_float_array(values):
    array = np.asarray(values)
    dtype = np.result_type(array.dtype, np.float32)
    return array.astype(dtype, copy=False)


def linear(values): return _to_float_array(values)
def relu(values):
    array = _to_float_array(values)
    return np.maximum(array, 0.0)

def softmax(values):
    array = _to_float_array(values)
    shifted = array - np.max(array, axis=-1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)

ACTIVATIONS = {
    "linear": linear,
    "relu": relu,
    "softmax": softmax,
}

def get_activation(activation):
    if activation is None:
        return linear
    if callable(activation):
        return activation

    normalized = activation.lower()
    if normalized not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {activation}")
    return ACTIVATIONS[normalized]