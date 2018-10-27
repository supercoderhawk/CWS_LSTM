import numpy as np


class ReLU(object):
    def __init__(self):
        self.name = 'ReLU'

    def encode(self, x):
        y = np.maximum(0.0, x)
        return (y)

    def bp(self, val):
        if (val <= 1e-5): return 0.0
        return 1.0


#### sigmoid
class Sigmoid(object):
    def __init__(self):
        self.name = 'Sigmoid'

    def encode(self, x):
        y = 1.0 / (1 + np.exp(-x))
        return (y)

    def bp(self, val):
        return val * (1.0 - val)


#### tanh
class Tanh(object):
    def __init__(self):
        self.name = 'Tanh'

    def encode(self, x):
        y = np.tanh(x)
        return (y)

    def bp(self, val):
        return 1.0 - val * val


class Linear(object):
    def __init__(self):
        self.name = 'Linear'

    def encode(self, x):
        return x

    def bp(self, val):
        return 1.0


class Layer(object):
    def __init__(self, rng, n_in, n_out, activation, W=None, b=None):
        multi = 1.0
        if (activation.name == 'Sigmoid'): multi = 4.0
        if W is None:
            W = np.asarray(rng.uniform(
                low=-multi * np.sqrt(6. / (n_in + n_out)),
                high=multi * np.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=np.float32)

        if b is None:
            b = np.zeros((n_out,), dtype=np.float32)

        self.W = W
        self.b = b
