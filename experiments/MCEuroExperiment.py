import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.utils.GBM import GBM
from blackscholes.mc.Euro import Euro
from blackscholes.mc.American import American
from utils.Experiment import MCEuroExperiment, MCEuroExperimentStd, MCAmerExperimentStd
import utils.Pickle as hdpPickle
import unittest
import numpy as np

class Test(unittest.TestCase):

    def test_amer_std(self):
        # although this is not a euro experiment...
        T = 1
        strike = 50
        asset_num = 1
        init_price_vec = 50*np.ones(asset_num)
        vol_vec = 0.5*np.ones(asset_num)
        ir = 0.05
        dividend_vec = np.zeros(asset_num)
        corr_mat = np.eye(asset_num)
        nTime = 365
        random_walk = GBM(T, nTime, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        def test_payoff(*l):
            return max(strike - np.sum(l), 0)
        opt = American(test_payoff, random_walk)
        MCAmerExperimentStd(10, 16, 30, opt)

    def test_amer(self):
        # although this is not a euro experiment...
        T = 1
        strike = 50
        asset_num = 1
        init_price_vec = 50*np.ones(asset_num)
        vol_vec = 0.5*np.ones(asset_num)
        ir = 0.05
        dividend_vec = np.zeros(asset_num)
        corr_mat = np.eye(asset_num)
        nTime = 365
        random_walk = GBM(T, nTime, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        def test_payoff(*l):
            return max(strike - np.sum(l), 0)
        opt = American(test_payoff, random_walk)
        analy = 8.723336355455928
        np.random.seed(1)
        result = MCEuroExperiment(analy, 10, 16, opt, "V1")
        hdpPickle.dump(result, 'MCAmer_1d.pickle')
        print(result)

    def test_std_6d(self):
        dim = 6
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        vol_vec = np.full(dim, vol)
        dividend_vec = np.full(dim, dividend)
        corr_mat = np.full((dim, dim), corr)
        np.fill_diagonal(corr_mat, 1)
        payoff_func = lambda x: np.maximum(strike - np.mean(x, axis=1), np.zeros(len(x)))
        random_walk = GBM(T, 400, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        opt = Euro(payoff_func, random_walk)
        MCEuroExperimentStd(10, 19, 500, opt)

    def test_conv_rate_6d(self):
        dim = 6
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        vol_vec = np.full(dim, vol)
        dividend_vec = np.full(dim, dividend)
        corr_mat = np.full((dim, dim), corr)
        np.fill_diagonal(corr_mat, 1)
        payoff_func = lambda x: np.maximum(strike - np.mean(x, axis=1), np.zeros(len(x)))
        random_walk = GBM(T, 400, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        opt = Euro(payoff_func, random_walk)
        analy = 1.50600
        np.random.seed(1)
        result = MCEuroExperiment(analy, 14, 21, opt, "V2")
        hdpPickle.dump(result, 'MCEuro_6d.pickle')
        print(result)
    
    
    def test_conv_rate_6d_control_sobol(self):
        dim = 6
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        vol_vec = np.full(dim, vol)
        dividend_vec = np.full(dim, dividend)
        corr_mat = np.full((dim, dim), corr)
        np.fill_diagonal(corr_mat, 1)
        payoff_func = lambda x: np.maximum(strike - np.mean(x, axis=1), np.zeros(len(x)))
        random_walk = GBM(T, 400, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        opt = Euro(payoff_func, random_walk)
        analy = 1.50600
        np.random.seed(1)
        result = MCEuroExperiment(analy, 14, 21, opt, "V8")
        hdpPickle.dump(result, 'MCEuro_6d_control_sobol.pickle')
        print(result)

    def test_conv_rate_6d_control(self):
        dim = 6
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        vol_vec = np.full(dim, vol)
        dividend_vec = np.full(dim, dividend)
        corr_mat = np.full((dim, dim), corr)
        np.fill_diagonal(corr_mat, 1)
        payoff_func = lambda x: np.maximum(strike - np.mean(x, axis=1), np.zeros(len(x)))
        random_walk = GBM(T, 400, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        opt = Euro(payoff_func, random_walk)
        analy = 1.50600
        np.random.seed(1)
        result = MCEuroExperiment(analy, 14, 21, opt, "V7")
        hdpPickle.dump(result, 'MCEuro_6d_control.pickle')
        print(result)

    def test_conv_rate_6d_antithetic(self):
        dim = 6
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        vol_vec = np.full(dim, vol)
        dividend_vec = np.full(dim, dividend)
        corr_mat = np.full((dim, dim), corr)
        np.fill_diagonal(corr_mat, 1)
        payoff_func = lambda x: np.maximum(strike - np.mean(x, axis=1), np.zeros(len(x)))
        random_walk = GBM(T, 400, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        opt = Euro(payoff_func, random_walk)
        analy = 1.50600
        np.random.seed(1)
        result = MCEuroExperiment(analy, 14, 21, opt, "V5")
        hdpPickle.dump(result, 'MCEuro_6d_Anti.pickle')
        print(result)

    def test_conv_rate_6d_Sobol(self):
        dim = 6
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        vol_vec = np.full(dim, vol)
        dividend_vec = np.full(dim, dividend)
        corr_mat = np.full((dim, dim), corr)
        np.fill_diagonal(corr_mat, 1)
        payoff_func = lambda x: np.maximum(strike - np.mean(x, axis=1), np.zeros(len(x)))
        random_walk = GBM(T, 400, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        opt = Euro(payoff_func, random_walk)
        analy = 1.50600
        np.random.seed(1)
        result = MCEuroExperiment(analy, 14, 21, opt, "V4")
        hdpPickle.dump(result, 'MCEuro_6d_Sobol.pickle')
        print(result)

    def test_conv_rate_4dGA(self):
        from scipy.stats.mstats import gmean
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
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
        result = MCEuroExperiment(analy, 14, 20, opt, "V2")
        hdpPickle.dump(result, 'MCEuro_4dGA.pickle')
        print(result)
    
    def test_conv_rate_4dGA_control(self):
        from scipy.stats.mstats import gmean
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
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
        result = MCEuroExperiment(analy, 14, 20, opt, "V7")
        hdpPickle.dump(result, 'MCEuro_4dGA_control.pickle')
        print(result)
    
    def test_conv_rate_4dGA_control_sobol(self):
        from scipy.stats.mstats import gmean
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
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
        result = MCEuroExperiment(analy, 14, 20, opt, "V8")
        hdpPickle.dump(result, 'MCEuro_4dGA_control_sobol.pickle')
        print(result)
    
    def test_conv_rate_4dGA_Sobol(self):
        from scipy.stats.mstats import gmean
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
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
        result = MCEuroExperiment(analy, 14, 20, opt, "V4")
        hdpPickle.dump(result, 'MCEuro_4dGA_Sobol.pickle')
        print(result)

    def test_conv_rate_4dGA_antithetic(self):
        from scipy.stats.mstats import gmean
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
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
        result = MCEuroExperiment(analy, 14, 20, opt, "V5")
        hdpPickle.dump(result, 'MCEuro_4dGA_Anti.pickle')
        print(result)

    def test_conv_rate_4dGA_anti_sol(self):
        from scipy.stats.mstats import gmean
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
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
        result = MCEuroExperiment(analy, 20, 24, opt, "V6")
        hdpPickle.dump(result, 'MCEuro_4dGA_AntiSol.pickle')
        print(result)

if __name__ == '__main__':
    unittest.main()