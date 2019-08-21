import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.utils.Regression import Monomial_Basis, Regression
import unittest
import numpy as np

class Test(unittest.TestCase):
    def setUp(self):
        pass
    
    def test_get_all_permutations(self):
        assert Monomial_Basis._get_all_permutations(2, 4) == [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 2, 0], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 2, 0, 0], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 1, 0, 0], [2, 0, 0, 0]]
    
    def test_basis_evaluate(self):
        basis = Monomial_Basis(1, 2)
        assert (list(map(lambda x: x.a_vec, basis.monomials)) == np.array([[0, 0], [0, 1], [1, 0]])).all()
        assert (basis.evaluate([2, 3]) == np.array([1, 3, 2])).all()

    def test_regression(self):
        r = Regression(np.array([[-1, -2], [3, 4], [5, 6], [4, 6], [3, 1]]), np.array([6, 7, 8, 9, 10]), 1, payoff_func=lambda x: np.sum(x))
        assert abs(r.evaluate([-1, -2]) - 7.878378378378377) < 0.00000000001
        assert abs(r.evaluate([3, 1]) - 9.594594594594595) < 0.00000000001
        


if __name__ == '__main__':
    unittest.main()