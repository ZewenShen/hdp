import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.fft.Conv import ConvEuro
from blackscholes.fft.GeometricAvg import Euro
from utils.Experiment import FFTConvExperiment
import utils.Pickle as hdpPickle
from blackscholes.utils.Analytical import GeometricAvg as AnalyticalGA

import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Test(unittest.TestCase):

    def test_all_sol_2d(self):
        T = 1
        dim = 2
        strike = 10
        init_price_vec = np.full(2, 50)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        euro = Euro(strike, init_price_vec, T, ir, vol, dividend, corr, 1)
        analy = AnalyticalGA(dim, init_price_vec, strike, T, ir, vol, dividend, corr).european_option_price()
        result = FFTConvExperiment(analy, 6, 7, euro, 10)
        print(result)
        grid, pri = euro.get_all_price()
        errors = []
        for i in range(len(pri)):
            ana = AnalyticalGA(dim, grid[i], strike, T, ir, vol, dividend, corr).european_option_price()
            errors.append(abs(ana - pri[i]))
        # print(errors)
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1) 
        x, y, errors = grid[:, 0], grid[:, 1], np.array(errors)
        ax.scatter(x[::3], y[::3], errors[::3], alpha=0.3)
        ax.set_xlabel('Spot Price of Asset 1')
        ax.set_ylabel('Spot Price of Asset 2')
        ax.axis('equal')
        plt.show()

    def test_conv_rate_4dGA(self):
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(4, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        euro = Euro(strike, init_price_vec, T, ir, vol, dividend, corr, 1)
        analy = AnalyticalGA(dim, init_price_vec, strike, T, ir, vol, dividend, corr).european_option_price()
        result = FFTConvExperiment(analy, 3, 7, euro, 10)
        hdpPickle.dump(result, 'FFTconv_rate_4dGA.pickle')
        print(result)

    def test_otm_4d(self):
        dim = 4
        T = 1
        strike = 80
        init_price_vec = np.full(dim, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        euro = Euro(strike, init_price_vec, T, ir, vol, dividend, corr, 1)
        analy = AnalyticalGA(dim, init_price_vec, strike, T, ir, vol, dividend, corr).european_option_price()
        result = FFTConvExperiment(analy, 3, 7, euro, 10)
        # hdpPickle.dump(result, 'FFTconv_rate_4dGA.pickle')
        print(result)


if __name__ == '__main__':
    unittest.main()