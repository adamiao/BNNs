"""
network.py
~~~~~~~~~~
Original code by Michael Nielsen:
http://neuralnetworksanddeeplearning.com/chap1.html

"A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network. Gradients are calculated
using backpropagation. Note that I have focused on making the code
simple, easily readable, and easily modifiable. It is not optimized,
and omits many desirable features." -Nielsen, M.

"Although I kept a lot of the original ideas of the code untouched, there were modifications that I made.
The purpose of these modifications were mainly to get a better understanding of the algorithm: see the
consequence of tweaks here and there like adding regularization parameters for example. Another thing I ended up doing
was to not have a final nonlinear layer at the output. This meant that I had to rearrange how the backpropagation
portion of the code. I've also removed some of the original methods while adding others to help me with what I needed.
It is not my intention to get any credit for the presented code below given it was originally devised by Nielsen."
- Damião, A.
"""

import random
import numpy as np


class Network:

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the respective layers of the network. For example,
        if the list was [2, 3, 1] then it would be a three-layer network, with the input layer containing 2 neurons,
        the (first) inner layer 3 neurons, and the output layer 1 neuron. The biases and weights for the network are
        initialized randomly, using a Gaussian distribution with mean 0, and variance 1."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    @staticmethod
    def sigmoid(z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        """Derivative of the sigmoid function."""
        return Network.sigmoid(z) * (1 - Network.sigmoid(z))

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input. If you have a network with sizes = [3, 2, 1] then the
        weights[0] will be of shape (how many nodes in the new layer are being targeted, how many nodes originated
        the signal)."""
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = Network.sigmoid(np.dot(w, a) + b)
        a = np.dot(self.weights[-1], a) + self.biases[-1]
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta=0.01, reg_eta=0.1, verbose=False):
        """Train the neural network using mini-batch stochastic gradient descent. The ``training_data`` is a
        list of tuples ``(x, y)`` representing the training inputs and the desired outputs. The other non-optional
        parameters are self-explanatory.  If ``test_data`` is provided then the network will be evaluated against
        the test data after each epoch, and partial progress printed out. This is useful for tracking progress, but
        slows things down substantially."""

        n = len(training_data)
        for j in range(epochs):
            shuffled_training_data = random.sample(training_data, n)
            mini_batches = [shuffled_training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, reg_eta)
            if verbose:
                if j % 50 == 0:
                    print("Epoch {0} complete".format(j))

    def update_mini_batch(self, mini_batch, eta, reg_eta):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single
        mini batch. The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta`` is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropagation(x, y, reg_eta)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - eta * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eta * nb for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, x, y, reg_eta):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the gradient for the cost function C_x. ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar to ``self.biases`` and ``self.weights``.
        Something of extreme importance that must be noted is that np.dot becomes matrix multiplication depending
        on the type of arrays you have. So be careful!"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Feedforward - Note that in the we will calculate the outputs prior to "entering" the activation function (z)
        # as well as the values being output by the activation function (activation) and put them in their respective
        # lists (zs) and (activations)

        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = Network.sigmoid(z)
            activations.append(activation)

        # Backward Pass - I have added a regularization parameter based on L2-norm of the parameters

        delta = zs[-1] - y
        nabla_b[-1] = delta - reg_eta*nabla_b[-1]
        nabla_w[-1] = np.dot(delta, activations[-2].transpose()) - reg_eta*nabla_w[-1]
        for idx in range(2, self.num_layers):
            z = zs[-idx]
            sigprime = Network.sigmoid_prime(z)
            delta = np.dot(self.weights[-idx+1].transpose(), delta) * sigprime
            nabla_b[-idx] = delta - reg_eta*nabla_b[-idx]
            nabla_w[-idx] = np.dot(delta, activations[-idx-1].transpose()) - reg_eta*nabla_w[-idx]

        return nabla_b, nabla_w

    @staticmethod
    def point_predictor(neural_network_list, list_of_gammas, x_input):
        y_shape = (neural_network_list[-1].sizes[-1], 1)
        prediction_point = np.zeros(y_shape)
        for idx, nn in enumerate(neural_network_list):
            prediction_point += list_of_gammas[idx] * nn.feedforward(x_input)
        return prediction_point

    @staticmethod
    def predictor(neural_network_list, list_of_gammas, training_x):
        x_shape = (len(training_x), len(training_x[0][0]))
        prediction_vector = np.zeros(x_shape)
        if len(list_of_gammas) == 0:
            for idx, x in enumerate(training_x):
                prediction_vector[idx] = neural_network_list[0].feedforward(x)
            return prediction_vector
        else:
            for idx, x in enumerate(training_x):
                for jdx, gamma in enumerate(list_of_gammas):
                    prediction_vector[idx] += gamma * neural_network_list[jdx].feedforward(x)
            return prediction_vector
