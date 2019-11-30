import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.utils.GBM import GBM
from blackscholes.mc.Euro import Euro
from utils.Experiment import MCEuroExperiment
import utils.Pickle as hdpPickle
import unittest
import numpy as np

class Test(unittest.TestCase):

    def test_conv_rate_4dGA(self):
        from scipy.stats.mstats import gmean
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(4, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        vol_vec = np.full(dim, vol)
        dividend_vec = np.full(dim, dividend)
        corr_mat = np.full((dim, dim), corr)
        np.fill_diagonal(corr_mat, 1)
        payoff_func = lambda x: np.maximum((gmean(x, axis=1) - strike), np.zeros(len(x)))
        random_walk = GBM(T, 400, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        opt = Euro(payoff_func, random_walk)
        analy = 2.165238512096621
        np.random.seed(1)
        result = MCEuroExperiment(analy, 14, 19, opt, "V2")
        hdpPickle.dump(result, 'MCEuro_4dGA.pickle')
        print(result)
    
    def test_conv_rate_4dGA_Sobol(self):
        from scipy.stats.mstats import gmean
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(4, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        vol_vec = np.full(dim, vol)
        dividend_vec = np.full(dim, dividend)
        corr_mat = np.full((dim, dim), corr)
        np.fill_diagonal(corr_mat, 1)
        payoff_func = lambda x: np.maximum((gmean(x, axis=1) - strike), np.zeros(len(x)))
        random_walk = GBM(T, 400, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        opt = Euro(payoff_func, random_walk)
        analy = 2.165238512096621
        np.random.seed(1)
        result = MCEuroExperiment(analy, 14, 19, opt, "V4")
        hdpPickle.dump(result, 'MCEuro_4dGA_Sobol.pickle')
        print(result)

    def test_conv_rate_4dGA_antithetic(self):
        from scipy.stats.mstats import gmean
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(4, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        vol_vec = np.full(dim, vol)
        dividend_vec = np.full(dim, dividend)
        corr_mat = np.full((dim, dim), corr)
        np.fill_diagonal(corr_mat, 1)
        payoff_func = lambda x: np.maximum((gmean(x, axis=1) - strike), np.zeros(len(x)))
        random_walk = GBM(T, 400, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        opt = Euro(payoff_func, random_walk)
        analy = 2.165238512096621
        np.random.seed(1)
        result = MCEuroExperiment(analy, 14, 19, opt, "V5")
        hdpPickle.dump(result, 'MCEuro_4dGA_Anti.pickle')
        print(result)


if __name__ == '__main__':
    unittest.main()