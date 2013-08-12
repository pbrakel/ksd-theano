import scipy
import theano
import theano.tensor as T
import numpy as np
from numpy.linalg import norm


class Model(object):
    """Contains symbolic variables required for training a differentiable
    model.

    The main purpose of this class is to compile a set of functions that are
    useful for second order optimization methods.
    """

    def __init__(self, p, inputs, s, costs):
        # useful data for reshaping
        self.shapes = [i.get_value().shape for i in p]
        self.sizes = map(np.prod, self.shapes)
        self.positions = np.cumsum([0] + self.sizes)[:-1]

        self.p = p
        self.inputs = inputs
        self.s = s
        self.costs = costs

        g = T.grad(costs[0], p)
        g = map(T.as_tensor_variable, g)  # for CudaNdarray
        self.f_gc = theano.function(inputs, g + costs)  # gradient computation
        self.f_cost = theano.function(inputs, costs)  # quick cost evaluation

        symbolic_types = T.scalar, T.vector, T.matrix, T.tensor3, T.tensor4

        coefficient = T.scalar()  # this is lambda*mu

        # this computes the product Gv = J'HJv (G is the Gauss-Newton matrix)
        v = [symbolic_types[len(i)]() for i in self.shapes]
        Jv = T.Rop(s, p, v)
        HJv = T.grad(T.sum(T.grad(costs[0], s)*Jv), s, consider_constant=[Jv])
        Gv = T.grad(T.sum(HJv*s), p, consider_constant=[HJv, Jv])
        Gv = map(T.as_tensor_variable, Gv)  # for CudaNdarray
        self.function_Gv = theano.function(inputs + v + [coefficient],
                                           Gv, givens={},
                                           on_unused_input='ignore')
        # compute J'sqrt(diag(H))v for jacobi preconditioner
        r = T.matrix()
        sqrt_Hv = T.sqrt(T.grad(T.sum(T.grad(costs[0], s)), s)) * r
        J_sqrt_Hv = T.Lop(s, p, sqrt_Hv)
        J_sqrt_Hv = map(T.as_tensor_variable, J_sqrt_Hv)  # for CudaNdarray

        self.function_J_sqrt_Hv = theano.function(inputs + [r],
                                           J_sqrt_Hv, givens={},
                                           on_unused_input='ignore')
        # compute Hv
        dp = T.grad(costs[0], p)
        total = 0
        for dp_, v_ in zip(dp, v):
            total += T.sum(dp_ * v_)

        Hv = T.grad(total, p)
        Hv = map(T.as_tensor_variable, Hv)  # for CudaNdarray
        self.function_Hv = theano.function(inputs + v + [coefficient], Hv,
                                           on_unused_input='ignore')

    def quick_cost(self, dataset, delta=0):
        if isinstance(delta, np.ndarray):
            delta = self.flat_to_list(delta)
        if type(delta) in (list, tuple):
            for i, d in zip(self.p, delta):
                i.set_value(i.get_value() + np.asarray(d, dtype='float32'))
        cost = np.sum([self.f_cost(*i)[0]
                        for i in dataset.iterate(update=False)]) / dataset.number_batches
        if type(delta) in (list, tuple):
            for i, d in zip(self.p, delta):
                i.set_value(i.get_value() - np.asarray(d, dtype='float32'))
        return cost

    def flat_to_list(self, vector):
        return [vector[position:position + size].reshape(shape)
                for shape, size, position in
                zip(self.shapes, self.sizes, self.positions)]

    def list_to_flat(self, l):
        return np.concatenate([i.flatten() for i in l])


def get_subspace(matvec, b, K=10, precon=None, d_prev=None):
    """Construct an orthogonal basis to the Krylov sumbpace K_k(A, b).

    matvec defines the product of A with a vector.

    The return values are the basis vectors and substitude Hessian in the
    subspace.
    """
    if precon is None:
        precon = lambda x: x
    if d_prev is None:
        d_prev = np.random.normal(0, 1, b.shape)
    subspace = np.zeros((b.shape[0], K + 1))

    p = precon(b)
    p = p / norm(p)
    subspace[:, 0] = p
    H = np.zeros((K + 1, K + 1))

    # Note that the subspace will have an additional vector.
    for k in range(K + 1):
        w = matvec(p)
        if k < K:
            u = precon(w)
        elif k == K:
            u = d_prev.copy()
        for j in range(k + 1):
            pj = subspace[:, j]
            H[k, j] = np.dot(w, pj)

            # TODO: first vector is not that orthogonal to the rest
            u = u - np.dot(u, pj) * pj

        if k <= K - 1:
            subspace[:, k + 1] = u / norm(u)
            p = subspace[:, k + 1]

    assert(not np.isnan(H).any())
    H += np.tril(H, -1).T
    return subspace, H


def get_optimizer_functions(L, P, get_cost, get_grad_cost, dataset):
    def f_obj(a):
        PCa = np.dot(P, scipy.linalg.solve_triangular(L.T, a))
        cost = get_cost(delta=PCa, dataset=dataset)
        return cost

    def f_prime(a):
        PCa = np.dot(P, scipy.linalg.solve_triangular(L.T, a))
        gradient = get_grad_cost(x=PCa, dataset=dataset)[0]
        CPg = scipy.linalg.solve_triangular(L, np.dot(P.T, gradient),
                                            lower=True)
        return CPg
    return f_obj, f_prime


def floor_matrix(H, eps):
    u, s, v = np.linalg.svd(H)
    max_s = np.max(s)
    s = np.asarray(map(lambda x: max(x, eps * max_s), s))
    H = np.dot(u * s, u.T)
    return H

def floor_vector(v, eps):
    max_v = np.max(v)
    return np.asarray(map(lambda x: max(x, eps * max_v), v))


def krylov_descent(model, gradient_dataset, matvec_dataset, bfgs_dataset,
                   iterations=100,
                   space_size=20,
                   matrix_type='G',
                   preconditioning=False,
                   maxfun=100):

    cost_list = []
    if matrix_type == 'G':
        matvec = model.function_Gv
    else:
        matvec = model.function_Hv

    def get_grad_cost(x=None, dataset=None):
        if x is not None:
            if isinstance(x, np.ndarray):
                x = model.flat_to_list(x)
            if type(x) in (list, tuple):
                for i, d in zip(model.p, x):
                    i.set_value(i.get_value() + np.asarray(d, dtype='float32'))

        if dataset is None:
            dataset = gradient_dataset

        gradient = np.zeros(sum(model.sizes),
                            dtype=theano.config.floatX)
        costs = []
        for inputs in dataset.iterate(update=False):
            result = model.f_gc(*inputs)
            gradient += model.list_to_flat(result[:len(model.p)])
            costs.append(result[len(model.p):])
        gradient /= dataset.number_batches
        # Restore original value
        if x is not None:
            if type(x) in (list, tuple):
                for i, d in zip(model.p, x):
                    i.set_value(i.get_value() - np.asarray(d, dtype='float32'))
        return gradient, costs

    def matvec_data(vector, dataset):
        v = model.flat_to_list(vector)
        result = 0
        # TODO: experimental line to see if different batches for different matvecs works
        for inputs in dataset.iterate(False):
            result += model.list_to_flat(
                matvec(*(inputs + v + [0]))) / dataset.number_batches
        return result

    eps = 1e-4

    for itn in range(iterations):
        # TODO: sample datasets a, b, c
        gradient, costs = get_grad_cost()
        b = -gradient
        cost_list.append(np.mean(costs, axis=0))

        print 'iteration:', itn, 'costs:', np.mean(costs, axis=0)
        if itn == 0:
            d_prev = np.zeros_like(gradient)
            d_prev[0] = 1

        matvec_dataset.update()
        bfgs_dataset.update()

        def matvec_wrapped(v):
            return matvec_data(np.asarray(v, dtype='float32'), matvec_dataset)

        print 'computing preconditioner...',
        if preconditioning == 'martens':
            M = 0
            counter = 0
            for inputs in matvec_dataset.iterate(update=False):
                for inp in inputs[0]:
                    M += model.list_to_flat(
                        model.f_gc(inp[None, :])[:len(model.p)])**2
                    counter += 1
            #M /= counter
            M = floor_vector(M, 1e-4)
            M **= -0.75  # actually 1/M
        elif preconditioning == 'jacobi':
            # function that computes J * sqrt(diag(H)) * v
            M = 0
            batch = 100
            counter = 0
            for tja in range(10):
                for inputs in matvec_dataset.iterate(update=False):
                    for i in range(0, len(inputs[0]), batch):
                        inp = inputs[0][i:i+batch]
                        v = np.asarray(np.sign(np.random.uniform(-1, 1, inp.shape)), dtype='float32')
                        M += model.list_to_flat(
                            model.function_J_sqrt_Hv(inp, v)[:len(model.p)])**2
                        counter += 1
            M = floor_vector(M, 1e-4)
            M **= .75
            M = 1 / M
        else:
            M = 1.0

        print 'done'

        def precon(v):
            return M * v


        print 'Computing subspace'
        P, H = get_subspace(matvec_wrapped, b,
                            space_size, precon, d_prev)
        H = floor_matrix(H, eps)
        L = np.linalg.cholesky(H)
        a = np.zeros(H.shape[0])

        f_obj, f_prime = get_optimizer_functions(L, P, model.quick_cost,
                get_grad_cost, bfgs_dataset)

        print 'Running BFGS...'
        results = scipy.optimize.fmin_bfgs(f_obj, a, f_prime,
                                             maxiter=maxfun, full_output=True)

        a_optimal = results[0]
        print 'gnorm', norm(results[2])
        d_prev = np.dot(P, scipy.linalg.solve_triangular(L.T, a_optimal))

        # update parameters
        for i, delta in zip(model.p, model.flat_to_list(d_prev)):
            i.set_value(i.get_value() + np.asarray(delta, dtype='float32'))
    cost_list.append(np.mean(costs, axis=0))
