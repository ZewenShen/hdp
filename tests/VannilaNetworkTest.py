import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.bsde.VannilaNetwork import SingleTimeStepNetwork
import tensorflow.keras.backend as K
import unittest
import numpy as np

class Test(unittest.TestCase):
    
    def setUp(self):
        self.single1 = SingleTimeStepNetwork(0.01, 5*np.ones(10), 3.3, 0.03, 0.1*np.ones(5-2))

    def test1(self):
        print(self.single1.model.input, self.single1.model.output)


if __name__ == '__main__':
    unittest.main()