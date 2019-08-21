import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from utils.PCA import PCA
import unittest
import numpy as np

class Test(unittest.TestCase):

    def test_reduce_dimension(self):
        X = np.array([[-1, -1, 0, 2, 0], [-2, 0, 0, 1, 1]])
        target = np.array([[-3/2**0.5, -1/2**0.5, 0, 3/2**0.5, 1/2**0.5]])
        assert np.sum(abs(PCA(X).reduce_dimension(1) - target)) < 0.000000000000001 
    
if __name__ == '__main__':
    unittest.main()