# Martens reports the mse even though the optimized quantity is the likelihood.
# Does he use multiply the error by .5?
import theano
import numpy as np
import theano.tensor as T
from optimizer import Model, krylov_descent
from autoencoder import DAE
import scipy.io as sio


class Dataset:

    def __init__(self, data, batch_size, number_batches):

        self.current_batch = 0
        self.batch_size = batch_size
        self.number_batches = number_batches
        self.items = []

        self.indices = range(len(data))
        np.random.shuffle(self.indices)

    def shuffle(self):
        np.random.shuffle(self.indices)

    def iterate(self, update=True):
        for i in xrange(self.number_batches):
            yield [data[self.indices[i * self.batch_size:(i + 1) * self.batch_size]]]
        if update: self.update()

    def update(self):
        self.shuffle()



DATA_PATH = '/home/pbrakel/Dropbox/Work/Repositories/ksd/krylov_descent/digs3pts_1.mat'

layer_sizes = [400, 200, 100, 50, 25, 6, 25, 50, 100, 200, 400]

layer_types = ['logistic',
               'logistic',
               'logistic',
               'logistic',
               'logistic',
               'linear',
               'logistic',
               'logistic',
               'logistic',
               'logistic',
               'logistic',
               'logistic']


input_dim = 28**2
batch_size = 1000
n_train = 18000
network = DAE(input_dim, layer_sizes, layer_types, sparse=True)

print 'Loading data...',
data_dict = sio.loadmat(DATA_PATH)
print 'Done'

data = np.asarray(data_dict['bdata'], dtype='float32')

train_data_array = data[:n_train]
val_data_array = data[n_train:]

train_data = Dataset(train_data_array, batch_size, n_train / batch_size)
matvec_data = Dataset(train_data_array, batch_size, 5)
bfgs_data = Dataset(train_data_array, batch_size, 5)
val_data = Dataset(val_data_array, batch_size, len(val_data_array) / batch_size)

x = T.matrix('x')
a, s = network.t_forward(x, return_linear=True)
mse_cost = T.mean((a - x)**2) * 28**2
cost = network.t_stable_ce(s, x).sum()
forward = theano.function([x], network.t_forward(x))

print 'Compiling functions...'
model = Model(network.parameters, [x], s, [cost, mse_cost])

print 'Starting training...'
krylov_descent(model, train_data, matvec_data, bfgs_data, iterations=150,
               space_size=80,
               matrix_type='G',
               maxfun=30,
               preconditioning='None')
