import unittest
import scipy
import numpy as np
from optimizer import get_subspace
from optimizer import get_optimizer_functions
from optimizer import floor_matrix


class TestSubspaceConstruction(unittest.TestCase):

    def run_with_small_matrix(self):
        a = np.random.normal(0, 1, (10, 10))
        A = np.dot(a, a.T)
        C = np.random.uniform(0.01, 1, 10)
        precon = lambda x: x * C
        matvec = lambda x: np.dot(A, x)
        b = np.random.normal(0, 1, 10)
        return get_subspace(matvec, b, 5, precon=precon)

    def test_orthonormality(self):
        eps = 1e-4
        P, H = self.run_with_small_matrix()
        PP = np.dot(P.T, P)
        diff = np.mean((np.eye(6) - PP)**2)
        self.assertLess(diff, eps)

    def test_H_symmetry(self):
        P, H = self.run_with_small_matrix()
        self.assertTrue(np.all(H == H.T))

    def test_H_eigenvalues(self):
        P, H = self.run_with_small_matrix()
        vals, vectors = np.linalg.eig(H)
        self.assertGreater(np.min(vals), 0)


class TestOptimizerFunctions(unittest.TestCase):

    def setUp(self):
        a = np.random.normal(0, 1, (10, 10))
        A = np.dot(a, a.T)
        matvec = lambda x: np.dot(A, x)
        b = np.random.normal(0, 1, 10)
        P, H = get_subspace(matvec, b, 5)
        L = np.linalg.cholesky(H)
        self.P = P
        self.H = H
        self.L = L

    def test_gradient(self):
        w = np.random.normal(0, 1, 10)
        d = np.random.normal(0, 1, 10)
        x0 = np.random.normal(0, 1, self.L.shape[0])
        def get_cost(delta, dataset):
            return np.log(1 + np.exp(-np.dot(w + delta, d)))
        def get_grad_cost(x, dataset):
            return (1 / (1 + np.exp(-np.dot(w + x, d))) - 1) * d, 0
        f_obj, f_prime = get_optimizer_functions(self.L, self.P, get_cost,
                                                 get_grad_cost, None)
        error = scipy.optimize.check_grad(f_obj, f_prime, x0)
        self.assertLess(error, 1e-5)


class TestFloorMatrix(unittest.TestCase):

    def test_pd_stays_the_same(self):
        """A positive definite matrix shouldn't change much."""
        a = np.random.normal(0, 1, (10, 10))
        A = np.dot(a, a.T)
        A_ = floor_matrix(A, 1e-4)
        self.assertLess(np.sum((A - A_)**2), 1e-8)
