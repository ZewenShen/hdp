import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.utils.GBM import GBM
from blackscholes.mc.American import American
import unittest
import numpy as np

class Test(unittest.TestCase):
    def setUp(self):
        strike = 303

        asset_num = 3
        init_price_vec = 100*np.ones(asset_num)
        vol_vec = 0.1*np.ones(asset_num)
        ir = 0.03
        dividend_vec = np.zeros(asset_num)
        corr_mat = np.eye(asset_num)
        random_walk = GBM(3, 3, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)

        def test_payoff(*l):
            return max(strike - np.sum(l), 0)

        self.opt1 = American(test_payoff, random_walk)
        
    def test_price1d(self):
        np.random.seed(444)
        self.opt1.price(3)

    def test_get_discounted_cash_flow(self):
        random_walk = GBM(3, 3, np.ones(1), 0.03, np.ones(1), np.zeros(1), np.eye(1))
        def test_payoff(*l):
            return max(3 - np.sum(l), 0)
        opt = American(test_payoff, random_walk)
        cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
        discounted = opt._get_discounted_cash_flow(2, cashflow_matrix, 3)
        assert sum(abs(discounted - np.array([2.9113366, 0, 1.94089107]))) < 0.00000001

        cashflow_matrix2 = np.array([[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
        discounted2 = opt._get_discounted_cash_flow(0, cashflow_matrix2, 3)
        assert sum(abs(discounted2 - np.array([2.8252936, 0, 1.82786237]))) < 0.00000001

if __name__ == '__main__':
    unittest.main()