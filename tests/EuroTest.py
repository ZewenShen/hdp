import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.utils.GBM import GBM
from blackscholes.utils.Analytical import Analytical_Sol
from blackscholes.mc.Euro import Euro
import unittest
import numpy as np

class Test(unittest.TestCase):
    def setUp(self):
        strike = 100

        asset_num = 1
        init_price_vec = 100*np.ones(asset_num)
        vol_vec = 0.1*np.ones(asset_num)
        ir = 0.03
        dividend_vec = np.zeros(asset_num)
        corr_mat = np.eye(asset_num)
        random_walk = GBM(1, 400, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        def test_payoff(*l):
            return max(strike - np.sum(l), 0)
        self.opt1 = Euro(test_payoff, random_walk)
        
        spot_price = init_price_vec[0]
        time_to_maturity = 1
        interest_rate = 0.03
        sigma = 0.1
        self.analytical1 = Analytical_Sol(spot_price, strike, time_to_maturity, interest_rate, sigma, dividend_yield=0)
        
    def test_price1d(self):
        np.random.seed(1)
        real_call, real_put = self.analytical1.european_option_price()
        approx_put = self.opt1.price(5000)
        assert abs(approx_put - 2.6101834050208175) < 0.00000000000001
        assert abs(approx_put-real_put)/real_put < 0.006187

if __name__ == '__main__':
    unittest.main()