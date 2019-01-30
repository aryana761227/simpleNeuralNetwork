import numpy as np
from ActivationFunction import sigmoid, sigmoid_derivative
from LossFunctions import mse_derivatives


class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 30)
        self.weights2 = np.random.rand(30, 20)
        self.weights3 = np.random.rand(20, self.y.shape[1])
        # real labels from dataset
        self.y = y
        # layer initialization
        self.layer1 = np.zeros(self.input.shape[0], self.weights1.shape[1])
        self.layer2 = np.zeros(self.input.shape[0], self.weights2.shape[1])
        # predicted labels from dataset
        self.output = np.zeros(self.y.shape)

    def feedforward(self):
        # feedforward(forwardpropagation) we will predict outputs
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        self.output = sigmoid(np.dot(self.layer2, self.weights3))

    def backpropagation(self):
        # we want find dloss/dw
        d_weights3 = np.dot(self.layer2.T, (mse_derivatives(self.y, self.output) * sigmoid_derivative(self.output)))
        d_weights2 = np.dot(self.layer1.T,
                            np.dot(mse_derivatives(self.y, self.output) * sigmoid_derivative(self.output),
                                   self.weights3.T) * sigmoid_derivative(self.layer2))
        d_weights1 = np.dot(self.input.T,
                            np.dot(np.dot(mse_derivatives(self.y, self.output) * sigmoid_derivative(self.output),
                                   self.weights3.T) * sigmoid_derivative(self.layer2), self.weights2.T) *
                            sigmoid_derivative(self.layer1))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.weights3 += d_weights3
