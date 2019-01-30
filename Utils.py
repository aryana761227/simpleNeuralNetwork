import numpy as np


class Math:

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.e ** (-x))

    @staticmethod
    def relu(x):
        return np.max(0, x)

    @staticmethod
    def leaky_relu(x, a=0.2):
        return np.max(a * x, x)

    @staticmethod
    def elu(x, a=0.2):
        if x >= 0:
            return x
        return a * (np.e ** x - 1)


