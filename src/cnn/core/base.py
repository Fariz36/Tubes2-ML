from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, name=None):
        self.name = name or self.__class__.__name__.lower()

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError

    def __call__(self, inputs):
        return self.forward(inputs)

    def get_config(self):
        return {"name": self.name}

class WeightedLayer(Layer, ABC):
    @abstractmethod
    def get_weights(self):
        raise NotImplementedError

    @abstractmethod
    def set_weights(self, weights):
        raise NotImplementedError