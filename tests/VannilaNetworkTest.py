import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.dl.keras.VannilaNetwork import SingleTimeStepNetwork
import unittest
import numpy as np

class Test(unittest.TestCase):
    
    def setUp(self):
        self.single1 = SingleTimeStepNetwork(0.01, 5*np.ones(10), 3.3)

if __name__ == '__main__':
    unittest.main()