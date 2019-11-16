import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.dgm.Euro import Euro1d, Euro
from utils.Domain import Domain1d, Sampler1dBoundary2Center, DomainNd, SamplerNd
from blackscholes.utils.Type import CallPutType
import unittest
import numpy as np
import tensorflow as tf

class Test(unittest.TestCase):

    def test_euro2d(self):
        dim = 2
        T = 1
        domain = DomainNd(np.array([[0, 3], [0, 3]]), T)
        vol_vec, ir, dividend_vec, strike = 0.1*np.ones(dim), 0.03, 0.01*np.ones(dim), 1
        payoff_func = lambda x: np.maximum(np.sum(x, axis=1) - strike, 0)
        corr_mat = 0.25 * np.ones((dim, dim))
        np.fill_diagonal(corr_mat, 1)
        solver = Euro(payoff_func, domain, vol_vec, ir, dividend_vec, corr_mat)
        solver.run(n_samples=2000, steps_per_sample=10)

    def test_euro1d(self):
        tf.random.set_random_seed(4)
        np.random.seed(4)
        T = 1
        domain = Domain1d(0, 2, T)
        vol, ir, dividend, strike = 0.1, 0.03, 0.01, 1
        solver = Euro1d(domain, vol, ir, dividend, strike, CallPutType.PUT)
        solver.run(n_samples=2000, steps_per_sample=10)

    def test_euro1d_bd2ct(self):
        tf.random.set_random_seed(4)
        np.random.seed(4)
        T = 1
        domain = Domain1d(0, 2, T)
        vol, ir, dividend, strike = 0.1, 0.03, 0.01, 1
        solver = Euro1d(domain, vol, ir, dividend, strike, CallPutType.PUT, sampler=Sampler1dBoundary2Center(domain))
        solver.run(n_samples=2000, steps_per_sample=10)
        
if __name__ == '__main__':
    unittest.main()