# sources:
# http://neuralnetworksanddeeplearning.com/chap1.html
# https://stevenmiller888.github.io/mind-how-to-build-a-neural-network/
# https://github.com/Ricky-N/NeuralNetwork-XOR/blob/master/xor.py

"""
wow i understand nothing
"""

import math
import numpy as np
import random


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.weights = [.01 * np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # finding number of weights and biases by multiplying neurons

    def sigmoid(self, data):
        return 1.0/(1.0+np.exp(-data))
        # basic sigmoid neuron

    def sigmoid_deriv(self, data):
        return sigmoid(data)*(1-sigmoid(data))
        # graphing slope of sigmoid function, chain rule

    def cost_deriv(self, activations, y):
        return (activations-y)


# --* begin processes *--

    # part 1: feed forward
    def forward_prop(self, data):
        for w, b in zip(self.weights, self.biases):
            data = self.sigmoid(np.dot(w, data) + b)
            # dot product of weights and inputs, plus biases
        return data

    # part 2: training network using SGD and backprop
    def gradient_descent(self, data, epochs, minibatch_size, lrate):
        n = len(data)
        for x in xrange(epochs):
            random.shuffle(data)
            minibatches = [data[y:y + minibatch_size]
                for y in xrange(0, n, minibatch_size)]
            for minibatch in minibatches:
                self.update(minibatch, lrate)

                """ I need to reformat 'update' and 'backprop'
                to make more intuitive sense. currently update calls
                the backprop function so I need
                to fix that somehow """

    # part 2a: back propagation
    def backprop(self, activation, y):

        # remove below? I'll test later
        #deriv_w = [np.zeros(w.shape) for w in self.weights]
        #deriv_b = [np.zeros(b.shape) for b in self.biases]

        activations = [activation]
        zs = []

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        delta = self.cost_deriv(activations[-1], y) * sigmoid_deriv(zs[-1])
        deriv_w[-1] = np.dot(delta, activations[-2].transpose())
        deriv_b[-1] = delta
        for layer in xrange(self.num_layers-2, 1, -1):
            #below doesn't seem necssary?? doesn't seem necessary
            #z = zs[-l]
            #sp = sigmoid_deriv(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_deriv(zs[layer])
            deriv_b[layer] = delta
            deriv_w[layer] = np.dot(delta, activations[layer-1].transpose())
        return (deriv_b, deriv_w)

    # part 2b: updating weights and biases
    def update(self, minibatch, lrate):
        # updating weights and biases
        for x, y in minibatch:
            output_w, output_b = self.backprop(x, y)
            deriv_w = [nb+dnb for nb, dnb in zip(deriv_w, partial_deriv_w)]
            deriv_b = [nw+dnw for nw, dnw in zip(deriv_b, partial_deriv_b)]

        # editing weights and biases
        self.weights = [weights-(lrate/len(minibatch))*new_weights
                        for weights, new_weights in zip(self.weights, deriv_w)]
        self.biases = [biases-(lrate/len(minibatch))*new_biases
                       for biases, new_biases in zip(self.biases, deriv_b)]



""" --* testing for now; will be edited later *-- """

training_data = np.random.randint(2, size=8)

net = Network([8, 3.0, 4])
data = net.gradient_descent(training_data, 30, 10, 3.0)
print('Weights: \n {}' .format(net.weights))
print('Biases: \n {}' .format(net.biases))
print('Outputs: \n {}' .format(data))
