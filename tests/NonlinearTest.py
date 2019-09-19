import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from utils.Domain import Domain1d
from burgers.pde.Nonlinear import Characteristic
import unittest
import numpy as np
# import matplotlib.pyplot as plt

class Test(unittest.TestCase):

    def test_Characteristic1(self):
        a, b, T = -6, 6, 1
        u0, du0 = lambda X, t: X, lambda X, t: 1*np.ones_like(X)
        domain = Domain1d(a, b, T, ic=u0)
        c = Characteristic(domain, du0)
        c.solve(200, 200)
        assert c.max_T == 1
        result = c.evaluate(np.linspace(-6, 6, 10), c.max_T)
        assert np.sum(abs((result - np.linspace(-6, 6, 10)/2))) < 2e-15

    def test_Characteristic2(self):
        a, b, T = -6, 6, 1
        u0, du0 = lambda X, t: -X, lambda X, t: -1*np.ones_like(X)
        domain = Domain1d(a, b, T, ic=u0)
        c = Characteristic(domain, du0)
        c.solve(200, 200)
        assert c.max_T == 1 - 1/200
        result = c.evaluate(np.linspace(-.2, .2, 10), T*0.95)
        assert abs(result[0] - 4) < 1e-14
        assert abs(result[-1] + 4) < 1e-14
        # plt.plot(np.linspace(-.2, .2, 10), result)
        # plt.savefig("s.jpg")

if __name__ == '__main__':
    unittest.main()