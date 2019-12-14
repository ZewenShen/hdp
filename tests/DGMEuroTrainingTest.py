import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.dgm.Euro import Euro1d, Euro, EuroV2, EuroV3
from utils.Domain import Domain1d, Sampler1dBoundary2Center, DomainNd, SamplerNd
from blackscholes.utils.Type import CallPutType
import unittest
import numpy as np
import tensorflow as tf

class Test(unittest.TestCase):

    def test_eurond1d(self):
        dim = 1
        T = 1
        domain = DomainNd(np.array([[0, 3]], dtype=np.float64), T)
        vol_vec, ir, dividend_vec, strike = 0.1*np.ones(dim, dtype=np.float64), 0.03, 0.01*np.ones(dim, dtype=np.float64), 1
        payoff_func = lambda x: tf.nn.relu(tf.math.reduce_sum(x, axis=1) - strike)
        corr_mat = 0.25 * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(corr_mat, 1)
        solver = Euro(payoff_func, domain, vol_vec, ir, dividend_vec, corr_mat)
        solver.run(n_samples=2000, steps_per_sample=1, n_interior=8, n_terminal=8)

    def test_euro2d(self):
        dim = 2
        T = 1
        domain = DomainNd(np.array([[0, 3], [0, 3]], dtype=np.float64), T)
        vol_vec, ir, dividend_vec, strike = 0.1*np.ones(dim, dtype=np.float64), 0.03, 0.01*np.ones(dim, dtype=np.float64), 1
        
        # payoff_func = lambda x: tf.nn.relu(tf.math.reduce_sum(x, axis=1) - strike)
        payoff_func = lambda x: tf.nn.relu(tf.pow(tf.math.reduce_prod(x, axis=1), 1/dim) - strike)

        corr_mat = 0.25 * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(corr_mat, 1)
        solver = Euro(payoff_func, domain, vol_vec, ir, dividend_vec, corr_mat)
        solver.run(n_samples=30000, steps_per_sample=1, saved_name="euro2d_geometric")

    def test_euroV2_2d(self):
        dim = 2
        T = 1.0
        domain = DomainNd(np.array([[0, 3], [0, 3]], dtype=np.float64), T)
        vol_vec, ir, dividend_vec, strike = 0.1*np.ones(dim, dtype=np.float64), 0.03, 0.01*np.ones(dim, dtype=np.float64), 1.0
        
        # payoff_func = lambda x: tf.nn.relu(tf.math.reduce_sum(x, axis=1) - strike)
        payoff_func = lambda x: tf.nn.relu(tf.pow(tf.math.reduce_prod(x, axis=1), 1/dim) - strike)

        corr_mat = 0.25 * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(corr_mat, 1.0)
        solver = EuroV2(payoff_func, domain, vol_vec, ir, dividend_vec, corr_mat)
        solver.run(n_samples=50000, steps_per_sample=1, saved_name="euroV2_2d_geometric")
    
    
    def test_euro4d(self):
        dim = 4
        T = 1.0
        domain = DomainNd(np.array([[20, 70], [20, 70], [20, 70], [20, 70]], dtype=np.float64), T)
        vol_vec, ir, dividend_vec, strike = 0.2*np.ones(dim, dtype=np.float64), 0.06, 0.04*np.ones(dim, dtype=np.float64), 40.0
        
        # payoff_func = lambda x: tf.nn.relu(tf.math.reduce_sum(x, axis=1) - strike)
        payoff_func = lambda x: tf.nn.relu(tf.pow(tf.math.reduce_prod(x, axis=1), 1/dim) - strike)

        corr_mat = 0.25 * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(corr_mat, 1.0)
        solver = Euro(payoff_func, domain, vol_vec, ir, dividend_vec, corr_mat)
        solver.run(n_samples=80000, steps_per_sample=1, saved_name="euro_4d_geometric")
    
    def test_euroV3_4d_dirichlet(self):
        dim = 4
        T = 1.0
        domain = DomainNd(np.array([[20, 60], [20, 60], [20, 60], [20, 60]], dtype=np.float64), T)
        vol_vec, ir, dividend_vec, stri = 0.2*np.ones(dim, dtype=np.float64), 0.06, 0.04*np.ones(dim, dtype=np.float64), 40.0
        payoff_func = lambda x: tf.nn.relu(tf.pow(tf.math.reduce_prod(x, axis=1), 1/dim) - stri)
        payoff_func.strike = stri

        corr_mat = 0.25 * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(corr_mat, 1.0)
        solver = EuroV3(payoff_func, domain, vol_vec, ir, dividend_vec, corr_mat)
        solver.run(n_samples=80000, steps_per_sample=1, saved_name="euroV3_4d_geometric_diri")

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