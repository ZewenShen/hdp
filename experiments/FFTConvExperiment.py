import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.fft.Conv import ConvEuro
from blackscholes.fft.GeometricAvg import Euro
from utils.Experiment import FFTConvExperiment
import utils.Pickle as hdpPickle

import unittest
import numpy as np


class Test(unittest.TestCase):

    def test_conv_rate_4dGA(self):
        T = 1
        strike = 40
        init_price_vec = np.full(4, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        euro = Euro(strike, init_price_vec, T, ir, vol, dividend, corr, 1)
        analy = 2.165238512096621
        result = FFTConvExperiment(analy, 3, 7, euro, 10)
        hdpPickle.dump(result, 'FFTconv_rate_4dGA.pickle')
        print(result)

    def test_otm(self):
        T = 1
        strike = 70
        init_price_vec = np.full(4, 40)
        vol = 0.2
        ir = 0.06
        dividend = 0.04
        corr = 0.25
        euro = Euro(strike, init_price_vec, T, ir, vol, dividend, corr, 1)
        # analy = 2.165238512096621
        # result = FFTConvExperiment(analy, 3, 7, euro, 10)
        # hdpPickle.dump(result, 'FFTconv_rate_4dGA.pickle')
        # print(result)


if __name__ == '__main__':
    unittest.main()