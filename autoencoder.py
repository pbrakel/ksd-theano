import theano
import numpy as np
import theano.tensor as T
from random import sample




sigmoid = lambda x: 1 / (1 + T.exp(-x))


class DAE(object):
    """Defines a multi autoencoder.

    layer_types can be either 'logistic' or 'linear' at the moment.
    """

    def __init__(self, input_dim, layer_sizes, layer_types, sparse=False):
        self.input_dim = input_dim
        self.layer_sizes = layer_sizes
        self.layer_types = layer_types
        self.weights = []
        self.biases = []
        self.weights.append(theano.shared(np.asarray(np.random.normal(0, 1, (input_dim, layer_sizes[0])), dtype='float32')))
        for i, j in zip(layer_sizes[:-1], layer_sizes[1:]):
            w = np.asarray(np.random.normal(0, 1, (i, j)), dtype='float32')
            self.weights.append(theano.shared(w))
        self.weights.append(theano.shared(np.asarray(np.random.normal(0, 1, (layer_sizes[-1], input_dim)), dtype='float32')))

        if sparse:
            for w in self.weights:
                w_ = w.get_value()
                m, n = w_.shape
                if m >= 15:
                    mask = np.zeros_like(w_)
                    for i in range(n):
                        indices = sample(range(m), 15)
                        mask[indices, i] = 1.0
                    w.set_value(w_ * mask)

        self.biases = [theano.shared(np.zeros(i, dtype='float32')) for i in layer_sizes]
        self.biases.append(theano.shared(np.zeros(input_dim, dtype='float32')))
        self.parameters = self.weights + self.biases


    def t_forward(self, x, return_linear=False):
        a = x
        for w, b, function in zip(self.weights, self.biases, self.layer_types):
            s = T.dot(a, w) + T.shape_padleft(b, 1)
            if function == 'logistic':
                a = sigmoid(s)
            else:
                a = s
        if return_linear:
            return a, s
        return a

    def t_ce(self, x, t):
        y = self.t_forward(x)
        return T.nnet.binary_crossentropy(y, t).mean()

    def t_stable_ce(self, s, t):
        # FIXME: I get nans if I use the mean
        return -(t * (s - T.log(1 + T.exp(s))) + (1 - t) *
                    (-s - T.log(1 + T.exp(-s))))
