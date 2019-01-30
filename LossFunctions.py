import numpy as np


def mse(arry, arry_hat):
    return np.square(arry - arry_hat)


def mae(arry, arry_hat):
    return np.abs(arry - arry_hat)


def mse_derivatives(arry, arry_hat):
    return 2 * (arry - arry_hat)


def mae_derivatives(arry, arry_hat):
    return np.sign(arry - arry_hat)