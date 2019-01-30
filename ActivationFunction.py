import numpy as np


def Sigmoid(arrx):
    arrx = -1 * arrx
    arrx = np.exp(arrx)
    arrx = arrx + 1
    arrx = 1./arrx
    return arrx


def Relu(arrx):
    arrx = np.maximum(0, arrx)
    return arrx


def LeakyRelu(arrx, a=0.2):
    arrx = np.maximum(a * arrx, arrx)
    return arrx

