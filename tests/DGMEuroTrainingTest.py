import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.dgm.Euro import Euro1d
from utils.Domain import Domain1d, Sampler1dBoundary2Center
from blackscholes.utils.Type import CallPutType
import unittest
import numpy as np
import tensorflow as tf

class Test(unittest.TestCase):

    def test_euro1d(self):
        tf.random.set_random_seed(4)
        np.random.seed(4)
        T = 1
        domain = Domain1d(0, 2, T)
        vol, ir, dividend, strike = 0.1, 0.03, 0.01, 1
        solver = Euro1d(domain, vol, ir, dividend, strike, CallPutType.PUT)
        solver.run(n_samples=2000, steps_per_sample=10)

    def test_euro1d_bd2ct(self):
        tf.random.set_random_seed(4)
        np.random.seed(4)
        T = 1
        domain = Domain1d(0, 2, T)
        vol, ir, dividend, strike = 0.1, 0.03, 0.01, 1
        solver = Euro1d(domain, vol, ir, dividend, strike, CallPutType.PUT, sampler=Sampler1dBoundary2Center(domain))
        solver.run(n_samples=2000, steps_per_sample=10)
        
if __name__ == '__main__':
    unittest.main()