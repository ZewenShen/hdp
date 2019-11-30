import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.utils.Analytical import Analytical_Sol, GeometricAvg

import unittest
import numpy as np


class Test(unittest.TestCase):

    def test_2d(self):
        T, dim = 1, 2
        vol_vec, ir, dividend, strike = 0.1*np.ones(dim, dtype=np.float32), 0.03, 0.01, 1
        corr_mat = 0.25 * np.ones((dim, dim), dtype=np.float32)
        np.fill_diagonal(corr_mat, 1)
        ga_price = GeometricAvg(dim, np.array([1, 1]), strike, T, ir, vol_vec, dividend, corr_mat).european_option_price()
        print(ga_price)

if __name__ == '__main__':
    unittest.main()