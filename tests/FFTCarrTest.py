import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.fft.Carr import CarrEuroCall1d
from blackscholes.utils.Analytical import Analytical_Sol as Ana1d
import unittest
import numpy as np

class Test(unittest.TestCase):

    def testCarr(self):
        S0 = 100.0  # index level
        K = 90.0  # option strike
        T = 1.0  # maturity date
        r = 0.05  # risk-less short rates
        sigma = 0.3  # volatility
        carr = CarrEuroCall1d(T, S0, r, sigma)
        f = carr.pricing_func(2**10, 0.0409061543436171)
        assert abs(f(K) - 19.69745057137447) < 1e-10
    
    def experimentCarrFullSol(self):
        S0 = 100.0  # index level
        K = 90.0  # option strike
        T = 1.0  # maturity date
        r = 0.05  # risk-less short rates
        sigma = 0.3  # volatility
        carr = CarrEuroCall1d(T, S0, r, sigma)
        f = carr.pricing_func(2**8, 0.0409061543436171)
        call, _ = Ana1d(S0, K, T, r, sigma).european_option_price()
        print(f(K), call)
        strike = carr.k_vec
        price = carr.price
        call, _ = Ana1d(S0, strike, T, r, sigma).european_option_price()
        print(price - call)
        import matplotlib.pyplot as plt
        print(len(strike))
        plt.plot(strike[110:130], (price-call)[110:130], marker=".")
        plt.show()


if __name__ == '__main__':
    unittest.main()