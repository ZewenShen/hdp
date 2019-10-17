import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.fft.Conv import ConvEuro
import unittest
import numpy as np

class Test(unittest.TestCase):

    def testRandZandG(self):
        k_vec = np.array([0, 1, 2, 3, 4])
        N_vec = 5 * np.ones(5)
        assert (ConvEuro.R(k_vec, N_vec) == np.array([0.5, 1, 1, 1, 0.5])).all()
        assert ConvEuro.Z(k_vec, N_vec) == 0.25
        assert ConvEuro.G(k_vec, N_vec) == 0.25

    def test_iterable_k_vec(self):
        N_vec = [3, 3]
        assert ConvEuro.iterable_k_vec(N_vec) == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

    def test_1dpricing(self):
        K = 1  # option strike
        T = 1.0  # maturity date
        r = 0.05  # risk-less short rates
        sigma = np.array([0.3])  # volatility
        dividend_vec = np.zeros(1)
        corr_mat = np.array([[1]])
        payoff_func = lambda x: np.maximum(x - K, 0)
        payoff_func.strike = K
        N = 512
        price = ConvEuro(payoff_func, T, r, sigma, dividend_vec, corr_mat).pricing_func(np.array([N]), np.array([(2*np.pi/N)**0.5]))
        print(price)
        

if __name__ == '__main__':
    unittest.main()