import matplotlib.pyplot as plt
import unittest
import numpy as np

class Test(unittest.TestCase):

    def test_6d_std(self):
        vanilla = np.array([7.91e-02, 5.17e-02, 3.66e-02, 2.45e-02, 1.78e-02,\
                            1.12e-02, 8.69e-03, 6.42e-03, 4.46e-03])
        antithetic = np.array([5.45e-02, 3.88e-02, 2.76e-02, 1.95e-02, 1.35e-02,\
                               9.44e-03, 6.82e-03, 4.64e-03, 3.28e-03])
        control = np.array([4.41e-02, 3.1e-02, 2.17e-02, 1.52e-02, 1.07e-02,\
                            7.64e-03, 5.28e-03, 3.17e-03, 2.59e-03])
        N = np.array([2**i for i in range(10, 19)])
        plt.plot(N, vanilla, '--o', N, antithetic, '--o', N, control, '--o')
        plt.legend(['vanilla MC', 'antithetic variates', 'control variates'])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('N')
        plt.ylabel('SD')
        plt.title('Standard Deviation of Different Variation Reduction Techniques')
        plt.savefig("std.png", dpi=1000)
        plt.show()
        

if __name__ == '__main__':
    unittest.main()