import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.dgm.Euro import Euro1d, Euro, EuroV2, EuroV3
from utils.Domain import Domain1d, DomainNd
from blackscholes.utils.Type import CallPutType
from blackscholes.utils.Analytical import Analytical_Sol, GeometricAvg
import unittest
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Test(unittest.TestCase):

    def test_euroV3_restore(self):
        model_name = "euroV3_4d_geometric_diri395405_80k"
        T, dim = 1, 4
        domain = DomainNd(np.array([[39.5, 40.5], [39.5, 40.5], [39.5, 40.5], [39.5, 40.5]], dtype=np.float64), T)
        vol_vec, ir, dividend_vec, strike = 0.1*np.ones(dim, dtype=np.float64), 0.03, 0.01*np.ones(dim, dtype=np.float64), 40.0
        corr_mat = 0.25 * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(corr_mat, 1)
        S = np.array([[40, 40, 40, 40], [40.5, 40.5, 40.5, 40.5], [39.5, 39.5, 39.5, 39.5],\
                      [40, 40, 40, 40], [40.5, 40.5, 40.5, 40.5], [39.5, 39.5, 39.5, 39.5]], dtype=np.float64)
        t = np.array([[0], [0], [0], [T], [T], [T]], dtype=np.float64)
        self.restore_helper_geometricV3(S, t, model_name, dim, T, domain, vol_vec, ir, dividend_vec, strike, corr_mat)

    def restore_helper_geometricV3(self, S, t, model_name, dim, T, domain, vol_vec, ir, dividend_vec, strike, corr_mat):
        payoff_func = lambda x: tf.nn.relu(tf.pow(tf.math.reduce_prod(x, axis=1), 1/dim) - strike)
        solver = EuroV3(payoff_func, domain, vol_vec, ir, dividend_vec, corr_mat)
        fitted = solver.restore(S, t, model_name)
        print(fitted)

    def test_euroV2_restore(self):
        model_name = "euroV2_2d_geometric"
        T, dim = 1, 2
        domain = DomainNd(np.array([[0, 3], [0, 3]], dtype=np.float64), T)
        vol_vec, ir, dividend_vec, strike = 0.1*np.ones(dim, dtype=np.float64), 0.03, 0.01*np.ones(dim, dtype=np.float64), 1
        corr_mat = 0.25 * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(corr_mat, 1)
        S = np.array([[1, 1], [2, 2]], dtype=np.float64)
        t = np.zeros(2, dtype=np.float64).reshape(-1, 1)
        self.restore_helper_geometricV2(S, t, model_name, dim, T, domain, vol_vec, ir, dividend_vec, strike, corr_mat)

    def restore_helper_geometricV2(self, S, t, model_name, dim, T, domain, vol_vec, ir, dividend_vec, strike, corr_mat):
        payoff_func = lambda x: tf.nn.relu(tf.pow(tf.math.reduce_prod(x, axis=1), 1/dim) - strike)
        solver = EuroV2(payoff_func, domain, vol_vec, ir, dividend_vec, corr_mat)
        fitted = solver.restore(S, t, model_name)
        print(fitted)
        # diff = abs(fitted-real)
        # plt.plot(Sanaly, real, alpha=0.7, label="Real")
        # plt.plot(Sanaly, fitted[0], alpha=0.7, label="Approx")
        # print("error: {}; max error:{}; mean error: {}".format(diff, np.max(diff), np.mean(diff)))
        # plt.legend()
        # plt.show()

    def test_euro_restore(self):
        model_name = "euro2d_geometric"
        T, dim = 1, 2
        domain = DomainNd(np.array([[0, 3], [0, 3]], dtype=np.float64), T)
        vol_vec, ir, dividend_vec, strike = 0.1*np.ones(dim, dtype=np.float64), 0.03, 0.01*np.ones(dim, dtype=np.float32), 1
        corr_mat = 0.25 * np.ones((dim, dim), dtype=np.float64)
        np.fill_diagonal(corr_mat, 1)
        S = np.array([[1, 1], [2, 2]], dtype=np.float64)
        t = np.zeros(2, dtype=np.float64).reshape(-1, 1)
        self.restore_helper_geometric(S, t, model_name, dim, T, domain, vol_vec, ir, dividend_vec, strike, corr_mat)

    def restore_helper_geometric(self, S, t, model_name, dim, T, domain, vol_vec, ir, dividend_vec, strike, corr_mat):
        payoff_func = lambda x: tf.nn.relu(tf.pow(tf.math.reduce_prod(x, axis=1), 1/dim) - strike)
        solver = Euro(payoff_func, domain, vol_vec, ir, dividend_vec, corr_mat)
        fitted = solver.restore(S, t, model_name)
        print(fitted)
        # diff = abs(fitted-real)
        # plt.plot(Sanaly, real, alpha=0.7, label="Real")
        # plt.plot(Sanaly, fitted[0], alpha=0.7, label="Approx")
        # print("error: {}; max error:{}; mean error: {}".format(diff, np.max(diff), np.mean(diff)))
        # plt.legend()
        # plt.show()

    def test_euro1d_restore(self):
        model_name = "Euro1d_20190914"

        tf.random.set_random_seed(4)
        np.random.seed(4)
        a, b, t, T = 0, 2, 0, 1
        nS = 21
        vol, ir, dividend, strike = 0.1, 0.03, 0.01, 1
        self.restore_helper1d(t, nS, model_name, a, b, T, vol, ir, dividend, strike, CallPutType.PUT)
    
    def restore_helper1d(self, t, nS, model_name, a, b, T, vol, ir, dividend, strike, cpType):
        domain = Domain1d(a, b, T)
        solver = Euro1d(domain, vol, ir, dividend, strike, cpType)
        Sanaly, S = np.linspace(domain.a, domain.b, nS), np.linspace(domain.a, domain.b, nS).reshape(-1, 1)
        fitted = solver.restore(S, t*np.ones_like(S), model_name)
        real_call, real_put = Analytical_Sol(Sanaly, strike, T-t, ir, vol, dividend_yield=dividend).european_option_price()
        if cpType == CallPutType.PUT:
            real = real_put
        elif cpType == CallPutType.CALL:
            real = real_call
        diff = abs(fitted-real)
        plt.plot(Sanaly, real, alpha=0.7, label="Real")
        plt.plot(Sanaly, fitted[0], alpha=0.7, label="Approx")
        print("error: {}; max error:{}; mean error: {}".format(diff, np.max(diff), np.mean(diff)))
        plt.legend()
        plt.show()

        
if __name__ == '__main__':
    unittest.main()