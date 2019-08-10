import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.utils.Regression import Monomial_Basis
import unittest
import numpy as np

class Test(unittest.TestCase):
    def setUp(self):
        self.basis = Monomial_Basis(2, 4)
    
    def test_get_all_permutations(self):
        assert self.basis._get_all_permutations(2, 4) == [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 2, 0], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [2, 0, 0, 0]]


if __name__ == '__main__':
    unittest.main()