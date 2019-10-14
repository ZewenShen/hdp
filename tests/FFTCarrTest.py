import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.fft.Carr import CarrEuroCall1d
import unittest
import numpy as np

class Test(unittest.TestCase):

    def testCarr(self):
        S0 = 100.0  # index level
        K = 90.0  # option strike
        T = 1.0  # maturity date
        r = 0.05  # risk-less short rates
        sigma = 0.3  # volatility
        f = CarrEuroCall1d(T, S0, r, sigma).pricing_func(2**10, 0.0409061543436171)
        assert abs(f(K) - 19.69745057137447) < 1e-10
        

if __name__ == '__main__':
    unittest.main()