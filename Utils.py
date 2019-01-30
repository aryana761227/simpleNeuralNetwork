import numpy as np


class Math:

    @staticmethod
    def sigmoid_f(x):
        return 1/(1 + np.e ** (-x))

    @staticmethod
    def relu_f(x):
        return np.max(0, x)

    @staticmethod
    def leaky_relu_f(x, a=0.2):
        return np.max(a * x, x)

    @staticmethod
    def elu_f(x, a=0.2):
        if x >= 0:
            return x
        return a * (np.e ** x - 1)


