import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.fft.Conv import ConvEuro
from blackscholes.fft.GeometricAvg import Euro as GAEuro
from blackscholes.fft.Basket import Euro as BasEuro, Digital
from utils.Experiment import FFTConvExperiment, FFTConvDeltaExperiment
import utils.Pickle as hdpPickle
from blackscholes.utils.Analytical import GeometricAvg as AnalyticalGA, Analytical_Sol as Ana1d

import unittest
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Test(unittest.TestCase):

    def test_delta_4d(self):
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(4, 40, dtype=np.float64)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        euro = GAEuro(strike, init_price_vec, T, ir, vol, dividend, corr, 1)
        analy = AnalyticalGA(dim, init_price_vec, strike, T, ir, vol, dividend, corr).delta()
        result = FFTConvDeltaExperiment(analy[0], 3, 8, euro, 10)
        hdpPickle.dump(result, 'FFTconv_delta_4dGA.pickle')
        print(result)

    def test_conv_rate_5dDig(self):
        dim = 6
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        euro = Digital(strike, init_price_vec, T, ir, vol, dividend, corr, 1)
        result = FFTConvExperiment(0.45, 2, 7, euro, 10, True)
        hdpPickle.dump(result, 'FFTconv_rate_5dDig.pickle')
        print(result)

    def test_conv_rate_6dBas(self):
        dim = 6
        T = 1
        strike = 40
        init_price_vec = np.full(dim, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        euro = BasEuro(strike, init_price_vec, T, ir, vol, dividend, corr, -1)
        result = FFTConvExperiment(1.50600, 2, 6, euro, 10, True)
        hdpPickle.dump(result, 'FFTconv_rate_6dBas.pickle')
        print(result)

    def test_1d_boundary_sol(self):
        T = 1
        spot = 50
        strike = 40
        ir = 0.1
        vol = 0.25
        call, _ = Ana1d(spot, strike, T, ir, vol).european_option_price()
        payoff_func = lambda x: np.maximum(x - strike, 0)
        conv = ConvEuro(payoff_func, spot*np.ones(1), T, ir, vol*np.ones(1), np.zeros(1), np.eye(1))
        result = FFTConvExperiment(call[0], 7, 8, conv, 10)
        print(result)
        grid, pri = conv.get_all_price()
        call, _ = Ana1d(grid.flatten(), strike, T, ir, vol).european_option_price()
        plt.plot(grid.flatten()[::5], call[::5], '--.', grid.flatten()[::5], pri[::5], '--.')
        plt.plot(spot, 14.296, 'o')
        plt.legend(['Analytical solution', 'Approximation', 'centre of the log space'])
        plt.savefig("conv1d.png", dpi=1500)
        plt.show()
        

    def test_all_sol_2d(self):
        T = 1
        dim = 2
        strike = 10
        init_price_vec = np.full(2, 50)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        euro = GAEuro(strike, init_price_vec, T, ir, vol, dividend, corr, 1)
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

    def test_all_sol_2d_3dplot(self):
        T = 1
        dim = 2
        strike = 10
        init_price_vec = np.full(2, 50)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        euro = GAEuro(strike, init_price_vec, T, ir, vol, dividend, corr, 1)
        analy = AnalyticalGA(dim, init_price_vec, strike, T, ir, vol, dividend, corr).european_option_price()
        result = FFTConvExperiment(analy, 6, 7, euro, 10)
        print(result)
        grid, pri = euro.get_all_price()
        errors = []
        for i in range(len(pri)):
            ana = AnalyticalGA(dim, grid[i], strike, T, ir, vol, dividend, corr).european_option_price()
            errors.append(pri[i] - ana)
        # print(errors)
        fig = plt.figure(figsize=(5, 5))
        ax = Axes3D(fig)
        x, y, errors = grid[:, 0], grid[:, 1], np.array(errors)
        ax.scatter(x[::3], y[::3], errors[::3], alpha=0.3, marker='.')
        ax.set_xlabel('Spot Price of Asset 1')
        ax.set_ylabel('Spot Price of Asset 2')
        ax.set_title('approximation - real')
        plt.show()

    def test_conv_rate_4dGA(self):
        dim = 4
        T = 1
        strike = 40
        init_price_vec = np.full(4, 40, dtype=np.float64)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        euro = GAEuro(strike, init_price_vec, T, ir, vol, dividend, corr, 1)
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
        euro = GAEuro(strike, init_price_vec, T, ir, vol, dividend, corr, 1)
        analy = AnalyticalGA(dim, init_price_vec, strike, T, ir, vol, dividend, corr).european_option_price()
        result = FFTConvExperiment(analy, 3, 7, euro, 10)
        # hdpPickle.dump(result, 'FFTconv_rate_4dGA.pickle')
        print(result)


if __name__ == '__main__':
    unittest.main()