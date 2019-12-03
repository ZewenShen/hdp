import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.utils.Analytical import Analytical_Sol, GeometricAvg

import unittest
import numpy as np


class Test(unittest.TestCase):

    def test_2d(self):
        T, dim = 1, 2
        vol_vec, ir, dividend, strike = 0.1*np.ones(dim, dtype=np.float64), 0.03, 0.01, 1
        corr_mat = 0.25 * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(corr_mat, 1)
        ga_price = GeometricAvg(dim, np.array([1, 1]), strike, T, ir, vol_vec, dividend, corr_mat).european_option_price()
        print(ga_price)
    
    def test_2d_greeks(self):
        T, dim = 1, 2
        vol_vec, ir, dividend, strike = 0.1*np.ones(dim, dtype=np.float64), 0.03, 0.01, 1
        corr_mat = 0.25 * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(corr_mat, 1)
        g = GeometricAvg(dim, np.array([20.0, 1.0]), strike, T, ir, vol_vec, dividend, corr_mat)
        delta = g.delta()
        gamma = g.gamma()
        print(delta, gamma)
    
    def test_4d_greeks(self):
        T, dim = 1, 4
        vol_vec, ir, dividend, strike = 0.2*np.ones(dim, dtype=np.float64), 0.06, 0.04, 40
        corr_mat = 0.25 * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(corr_mat, 1)
        g = GeometricAvg(dim, np.array([40.0, 40.0, 40.0, 40.0]), strike, T, ir, vol_vec, dividend, corr_mat)
        val = g.european_option_price()
        delta = g.delta()
        gamma = g.gamma()
        print(val, delta, gamma)

if __name__ == '__main__':
    unittest.main()