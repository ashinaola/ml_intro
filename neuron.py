import numpy as np
from PyQt5.QtGui.QRawFont import weight


# define function for the activation function (1/1-e^(-x))
# input: x -> dot product of inputs and weights
def activation_function(self, x):
    return 1 / (1 + np.exp(-x))

'''
class neuron
weights
bias
neuron()
forwarding(vector) -> dot product(x1,x2), put through act func
'''
class Neuron:
    def __init__(self, weight, bias):
        self.weights = weight
        self.bias = bias

    def forwarding(self, inputs):
        x = np.dot(self.weights, inputs)