import numpy as np

class Sequential:
    def __init__(self, layers=None, name="sequential"):
        self.name = name
        self.layers = list(layers or [])

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        outputs = np.asarray(inputs)
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs

    def __call__(self, inputs):
        return self.forward(inputs)

    def predict(self, inputs, batch_size=None):
        array = np.asarray(inputs)
        if batch_size is None or array.ndim == 3:
            return self.forward(array)

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        predictions = []
        for start in range(0, array.shape[0], batch_size):
            batch = array[start : start + batch_size]
            predictions.append(self.forward(batch))
        return np.concatenate(predictions, axis=0)