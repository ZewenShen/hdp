import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.fft.Conv import ConvEuro, ConvEuro1d
from blackscholes.fft.GeometricAvg import Euro

import unittest
import numpy as np

class Test(unittest.TestCase):

    def test_1dpricing(self):
        K = 110  # option strike
        T = 0.1  # maturity date
        r = 0.1  # risk-less short rates
        dividend = 0
        S0 = 100
        sigma = 0.25  # volatility
        payoff_func = lambda x: np.maximum(K - x, 0)
        payoff_func.strike = K
        price = ConvEuro1d(payoff_func, S0, T, r, sigma, dividend).price(9)
        assert abs(price - 9.49503156495164) < 1e-10

    def test_conv1d(self):
        K = 110  # option strike
        T = 0.1  # maturity date
        r = 0.1  # risk-less short rates
        dividend_vec = np.array([0])
        S0_vec = np.array([100])
        sigma = np.array([0.25])  # volatility
        payoff_func = lambda x: np.maximum(K - x, 0)
        corr_mat = np.array([[1]])
        euro1d = ConvEuro(payoff_func, S0_vec, T, r, sigma, dividend_vec, corr_mat)
        price = euro1d.price(np.array([9]))
        assert abs(price - 9.495345187808168) < 1e-10
        delta, gamma = euro1d.greeks()
        assert sum(abs(delta - np.array([-0.85056763]))) < 1e-8
        assert sum(abs(gamma - np.array([0.02939849]))) < 1e-8
    
    def test_conv2d(self):
        T = 182/365
        strike = 50
        asset_num = 2
        init_price_vec = np.array([110, 60])
        vol_vec = np.array([0.4, 0.2])
        ir = 0.1
        dividend_vec = 0*np.ones(asset_num)
        corr_mat = np.eye(asset_num)
        corr_mat[0, 1] = 0.4
        corr_mat[1, 0] = 0.4
        def test_payoff(l):
            return np.maximum(l[:, 0] - l[:, 1] - strike, 0)
        euro2d = ConvEuro(test_payoff, init_price_vec, T, ir, vol_vec, dividend_vec, corr_mat)
        price = euro2d.price(np.array([6, 6]))
        assert abs(price - 12.549345239160479) < 1e-10
        real = 12.5583468
        # print(abs(price-real) / real)
        assert abs(price-real) / real < 0.00071678
        delta, gamma = euro2d.greeks()
        assert sum(abs(delta - np.array([0.57780525, -0.48478483]))) < 1e-8
        assert sum(abs(gamma - np.array([0.01325285, 0.01443476]))) < 1e-8

        """
        Test 2: european 2d spread put
        """
        def test_payoff2(l):
            return np.maximum(-l[:, 0] + l[:, 1] + strike, 0)
        euro2d2 = ConvEuro(test_payoff2, init_price_vec, T, ir, vol_vec, dividend_vec, corr_mat)
        price2 = euro2d2.price(np.array([7, 7]))
        mc_approximation = 10.108531893795202
        assert abs(price2-mc_approximation) / mc_approximation < 0.00164
        delta2, gamma2 = euro2d2.greeks()
        assert sum(abs(delta2 - np.array([-0.41634229, 0.51450032]))) < 1e-8
        assert sum(abs(gamma2 - np.array([0.01348562, 0.01425442]))) < 1e-8
        
    def test_geometric_avg_4d(self):
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        euro = Euro(strike, init_price_vec, T, ir, vol, dividend, corr, 1)
        # import profile
        # profile.run('Euro(40, np.full(4, 40), 1, 0.06, 0.2, 0.04, 0.25, 1).price(6*np.ones(4, dtype=int))')
        price = euro.price(5*np.ones(dim, dtype=int), 10)
        assert abs(price - 2.1443240208017147) < 1e-10
        # print(price)
        greeks = euro.greeks()
        print(greeks)

if __name__ == '__main__':
    unittest.main()