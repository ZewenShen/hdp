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
        def test_payoff(l):
            tmp = np.sum(l, axis=1)
            return np.maximum(strike - tmp, np.zeros_like(tmp))
        self.opt1 = Euro(test_payoff, random_walk)
        
        spot_price = init_price_vec[0]
        time_to_maturity = 1
        interest_rate = 0.03
        sigma = 0.1
        self.analytical1 = Analytical_Sol(spot_price, strike, time_to_maturity, interest_rate, sigma, dividend_yield=0)
    
    def test_sobol(self):
        _, real_put = self.analytical1.european_option_price()
        approx_put = self.opt1.priceV4(10000)
        assert abs(approx_put - 2.6263259615779786) < 0.00000000000001
        assert abs(approx_put-real_put)/real_put < 3.98046746e-05
    
    def test_nd_control_variates(self):
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
        np.random.seed(1)
        price = opt.priceV7(100000)
        assert abs(price - 2.16043821457437) < 1e-10

    def test_geometric_avg_4d(self):
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
        np.random.seed(1)
        price = opt.priceV2(100000) # "real": 2.164959740690803
        assert abs(price - 2.1654452369352635) < 1e-10
        assert (price - 2.164959740690803)/2.164959740690803 < 0.0002243

    def test_correlated_pricing(self):
        strike = 50
        asset_num = 2
        init_price_vec = np.array([110, 60])
        vol_vec = np.array([0.4, 0.2])
        ir = 0.1
        dividend_vec = 0*np.ones(asset_num)
        corr_mat = np.eye(asset_num)
        corr_mat[0, 1] = 0.4
        corr_mat[1, 0] = 0.4
        random_walk = GBM(182/365, 400, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        def test_payoff(l):
            return np.maximum(l[:, 0] - l[:, 1] - strike, np.zeros(len(l)))
        opt = Euro(test_payoff, random_walk)
        np.random.seed(1)
        callV2 = opt.priceV2(1000000)
        real_call = 12.5583468
        assert abs(callV2 - 12.566682253085943) < 0.00000000000001
        assert abs(callV2 - real_call)/real_call < 0.00066374
        np.random.seed(1)
        callV3 = opt.priceV3(20000)
        assert abs(callV3 - 12.586752483453562) < 0.00000000000001
        assert abs(callV3 - real_call)/real_call < 0.0022619

        """
        Test 2: european 2d spread put
        """
        np.random.seed(1)
        def test_payoff2(l):
            return np.maximum(-l[:, 0] + l[:, 1] + strike, np.zeros(len(l)))
        opt = Euro(test_payoff2, random_walk)
        spreadput2d = opt.priceV2(500000)
        assert abs(spreadput2d - 10.108531893795202) < 1e-10 

    def test_price1d(self):
        np.random.seed(1)
        _, real_put = self.analytical1.european_option_price()
        approx_put = self.opt1.price(5000)
        assert abs(approx_put - 2.6101834050208175) < 0.00000000000001
        assert abs(approx_put-real_put)/real_put < 0.006187

    def test_price1d_V2(self):
        np.random.seed(1)
        _, real_put = self.analytical1.european_option_price()
        approx_put = self.opt1.priceV2(300000)
        assert abs(approx_put - 2.61594175018011) < 0.00000000000001
        assert abs(approx_put-real_put)/real_put < 0.003994
    
    def test_price_antithetic_variates(self):
        np.random.seed(1)
        _, real_put = self.analytical1.european_option_price()
        approx_put = self.opt1.price_antithetic_variates(5000)
        assert abs(approx_put - 2.631103908508011) < 0.00000000000001
        assert abs(approx_put-real_put)/real_put < 0.00178

    def test_price1d_control_variates(self):
        strike = 45
        asset_num = 1
        init_price_vec = 50*np.ones(asset_num)
        vol_vec = 0.3*np.ones(asset_num)
        ir = 0.05
        dividend_vec = np.zeros(asset_num)
        corr_mat = np.eye(asset_num)
        time_to_maturity = 0.25
        random_walk = GBM(time_to_maturity, 100, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        analytical2 = Analytical_Sol(init_price_vec[0], strike, time_to_maturity, ir, vol_vec[0], dividend_yield=0)
        def test_payoff(l):
            return np.maximum(np.sum(l, axis=1)-strike, np.zeros(len(l)))
        opt2 = Euro(test_payoff, random_walk)

        real_call, _ = analytical2.european_option_price()
        np.random.seed(1)
        approx_call = opt2.price1d_control_variates(1000)
        np.random.seed(1)
        approx_call2 = opt2.price(1000)
        assert abs(approx_call - 6.412754547048265) < 0.00000000000001
        assert abs(approx_call-real_call)/real_call < 0.0025422
        assert abs(abs(approx_call2-real_call)/real_call) < 0.04899

    def test_price_importance_sampling(self):
        strike = 80
        asset_num = 1
        init_price_vec = 50*np.ones(asset_num)
        vol_vec = 0.2*np.ones(asset_num)
        ir = 0.03
        dividend_vec = np.zeros(asset_num)
        corr_mat = np.eye(asset_num)
        time_to_maturity = 1
        random_walk = GBM(time_to_maturity, 100, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
        analytical2 = Analytical_Sol(init_price_vec[0], strike, time_to_maturity, ir, vol_vec[0], dividend_yield=0)
        def test_payoff(l):
            return np.maximum(np.sum(l, axis=1)-strike, np.zeros(len(l)))
        test_payoff.strike = strike

        opt2 = Euro(test_payoff, random_walk)
        real_call, _ = analytical2.european_option_price()
        np.random.seed(1)
        approx_call = opt2.price_importance_sampling(10000)
        np.random.seed(1)
        weak_approx_call = opt2.priceV2(10000)
        assert abs(approx_call - 0.06082838151186516) < 0.00000000000001
        assert abs(approx_call-real_call)/real_call < 0.00297664353761824
        assert abs(weak_approx_call-real_call)/real_call < 0.1362349660567213

if __name__ == '__main__':
    unittest.main()