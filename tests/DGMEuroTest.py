import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.dgm.Euro import Euro1d
from blackscholes.utils.Domain import Domain1d
from blackscholes.utils.Type import CallPutType
import unittest
import numpy as np
import tensorflow as tf

class Test(unittest.TestCase):

    def test_euro1d(self):
        tf.random.set_random_seed(4)
        T = 0.01
        domain = Domain1d(0, 6, T)
        vol, ir, dividend, strike = 0.1, 0.03, 0.01, 1
        solver = Euro1d(domain, vol, ir, dividend, strike, CallPutType.PUT)
        solver.run(n_samples=200, steps_per_sample=10)
        S_interior_tnsr = tf.placeholder(tf.float32, [None, 1])
        t_interior_tnsr = tf.placeholder(tf.float32, [None, 1])
        V = solver.model(S_interior_tnsr, t_interior_tnsr)
        with tf.Session() as sess:
            S_plot = np.linspace(0, 6, 50).reshape(-1, 1)
            t_plot = np.zeros_like(S_plot)
            sess.run(tf.global_variables_initializer())
            fitted_optionValue = sess.run([V], feed_dict= {S_interior_tnsr: S_plot, t_interior_tnsr: t_plot})
            print(fitted_optionValue)
            S_plot = np.linspace(0, 6, 50).reshape(-1, 1)
            t_plot = T*np.ones_like(S_plot)
            sess.run(tf.global_variables_initializer())
            fitted_optionValue = sess.run([V], feed_dict= {S_interior_tnsr: S_plot, t_interior_tnsr: t_plot})
            print(fitted_optionValue)
if __name__ == '__main__':
    unittest.main()