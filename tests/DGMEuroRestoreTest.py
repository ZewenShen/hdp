import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.dgm.Euro import Euro1d
from utils.Domain import Domain1d
from blackscholes.utils.Type import CallPutType
from blackscholes.utils.Analytical import Analytical_Sol
import unittest
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Test(unittest.TestCase):

    def test_euro1d_restore(self):
        model_name = "Euro1d_20190914"

        tf.random.set_random_seed(4)
        np.random.seed(4)
        a, b, t, T = 0, 2, 0, 1
        nS = 21
        vol, ir, dividend, strike = 0.1, 0.03, 0.01, 1
        self.restore_helper(t, nS, model_name, a, b, T, vol, ir, dividend, strike, CallPutType.PUT)
    
    def restore_helper(self, t, nS, model_name, a, b, T, vol, ir, dividend, strike, cpType):
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