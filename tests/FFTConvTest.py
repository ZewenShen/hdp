import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.fft.Conv import ConvEuro
import unittest
import numpy as np

class Test(unittest.TestCase):

    def testRandZandG(self):
        k_vec = np.array([0, 1, 2, 3, 4])
        N_vec = 5 * np.ones(5)
        assert (ConvEuro.R(k_vec, N_vec) == np.array([0.5, 1, 1, 1, 0.5])).all()
        assert ConvEuro.Z(k_vec, N_vec) == 0.25
        assert ConvEuro.G(k_vec, N_vec) == 0.25
        

if __name__ == '__main__':
    unittest.main()