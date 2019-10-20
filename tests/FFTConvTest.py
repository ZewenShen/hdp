import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.fft.Conv import ConvEuro, ConvEuro1d
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
        K = 110  # option strike
        T = 0.1  # maturity date
        r = 0.1  # risk-less short rates
        dividend = 0
        S0 = 100
        sigma = 0.25  # volatility
        payoff_func = lambda x: np.maximum(K - x, 0)
        payoff_func.strike = K
        price = ConvEuro1d(payoff_func, S0, T, r, sigma, dividend).pricing_func(9)
        assert abs(price - 9.49503156495164) < 1e-10

    def test_conv(self):
        K = 110  # option strike
        T = 0.1  # maturity date
        r = 0.1  # risk-less short rates
        dividend_vec = np.array([0])
        S0_vec = np.array([100])
        sigma = np.array([0.25])  # volatility
        payoff_func = lambda x: np.maximum(K - x, 0)
        payoff_func.strike = K
        corr_mat = np.array([[1]])
        a = ConvEuro(payoff_func, S0_vec, T, r, sigma, dividend_vec, corr_mat)

        price = a.pricing_func(np.array([9]))
        print(price)

if __name__ == '__main__':
    unittest.main()