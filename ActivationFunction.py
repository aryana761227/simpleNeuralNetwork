import numpy as np


def sigmoid(arrx):
    arrx = -1 * arrx
    arrx = np.exp(arrx)
    arrx = arrx + 1
    arrx = 1./arrx
    return arrx


def relu(arrx):
    arrx = np.maximum(0, arrx)
    return arrx


def leaky_relu(arrx, a=0.2):
    arrx = np.maximum(a * arrx, arrx)
    return arrx


def tanh(arrx):
    arrx = np.tanh(arrx)
    return arrx


def sigmoid_derivative(arrz):
    return arrz - np.square(arrz)


def relu_derivative(arrz):
    # 1 if z > 0 and 0 if z < 0 and in formal z = 0 --> 0
    return relu(np.sign(arrz))


def leaky_relu_derivative(arrz, a=0.2):
    # 1 if z > 0 and a if z < 0 and in formal z = 0 --> 0 and a must be 0 < a < 1
    t1 = relu(np.sign(arrz))
    t2 = a * relu(np.sign(-arrz))
    return t1 + t2


def tanh_derivative(arrz):
    return 1 - np.square(arrz)
