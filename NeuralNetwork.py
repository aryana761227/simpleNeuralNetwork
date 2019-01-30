import numpy as np
from ActivationFunction import sigmoid, sigmoid_derivative
from LossFunctions import mse_derivatives


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.layer1 = self.layer1 = sigmoid(np.dot(self.input, self.weights1))

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backpropogation(self):
        # we want find dloss/dw
        d_weights2 = np.dot(self.layer1.T, (mse_derivatives(self.y, self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T, np.dot(mse_derivatives(self.y, self.output) * sigmoid_derivative(self.output),
                                                 self.weights2.T) * sigmoid_derivative(self.layer1))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
