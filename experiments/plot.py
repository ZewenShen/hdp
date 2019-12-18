import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
import matplotlib.pyplot as plt
import unittest
import numpy as np
import utils.Pickle as pickle

class Test(unittest.TestCase):

    def test_plot_loss(self):
        loss_vec = pickle.load("BlackScholes_AmericanPut1217_lossvec.pickle")
        l1_vec = pickle.load("BlackScholes_AmericanPut1217_l1vec.pickle")
        l2_vec = pickle.load("BlackScholes_AmericanPut1217_l2vec.pickle")
        l3_vec = pickle.load("BlackScholes_AmericanPut1217_l3vec.pickle")
        l4_vec = pickle.load("BlackScholes_AmericanPut1217_l4vec.pickle")
        domain = np.arange(1, 3001)
        all_vec = [domain, loss_vec, l1_vec, l2_vec, l3_vec, l4_vec]
        for i in range(len(all_vec)):
            all_vec[i] = all_vec[i][::50]
        plt.plot(all_vec[0], all_vec[1], linewidth=2)
        for i in range(2, len(all_vec)):
            plt.plot(all_vec[0], all_vec[i], '--')
        plt.legend(['Sum', 'L1', 'L2', 'L3', 'L4'])
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        # plt.plot(domain, loss_vec, domain, l1_vec, domain, l2_vec, domain, l3_vec, domain, l4_vec)
        plt.yscale('log')
        plt.savefig("loss.png", dpi=1000)


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
    
    def test_6d_conv(self):
        N = np.array([4**6, 8**6, 16**6, 32**6])
        err = np.array([1.3e-01, 3.5e-03, 3.4e-04, 1.7e-05])
        plt.plot(N, err, '--o')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Number of grid points')
        plt.ylabel('Relative error')
        plt.legend(['FFT Conv'])
        # plt.title("FFT Conv method's accuracy versus number of grid points")
        plt.savefig("conv6d.png", dpi=1500)
        plt.show()

    def test_6d_acc_vs_time(self):
        vanilla_t = np.array([0.67, 1.33, 2.72, 5.52, 22.17])
        vanilla_e = np.array([7.0e-03, 1.1e-02, 5.4e-03, 2.4e-03, 7.7e-04])
        anti_t = np.array([0.21, 0.37, 0.65, 1.31, 2.63, 5.31, 10.65, 21.34])
        anti_e = np.array([9.6e-03, 4.5e-03, 4.7e-03, 7.6e-03, 5.1e-03, 1.6e-04, 2.9e-03, 3.4e-04])
        ctrl_t = np.array([0.57, 1.00, 2.01, 4.01, 16.25])
        ctrl_e = np.array([7.6e-03, 9.4e-03, 6.3e-03, 4.1e-04, 4.6e-04])
        sobol_t = np.array([0.76, 1.49, 2.96, 5.97, 11.94, 24.08])
        sobol_e = np.array([2.7e-04, 5.9e-04, 8.3e-04, 2.2e-04, 5.9e-05, 4.6e-06])
        sobctrl_t = np.array([0.86, 1.52, 3.06, 6.09, 12.31, 24.73])
        sobctrl_e = np.array([1.8e-04, 8.5e-04, 9.6e-04, 2.9e-04, 1.9e-05, 2.4e-05])
        fft_t = np.array([0.19, 16.4])
        fft_e = np.array([3.5e-03, 3.4e-04])
        # plt.plot(vanilla_t, vanilla_e, '--o', anti_t, anti_e, '--o', ctrl_t, ctrl_e, '--o',\
        #          sobol_t, sobol_e, '--o', sobctrl_t, sobctrl_e, '--o', fft_t, fft_e, '--o')
        plt.plot(vanilla_t, vanilla_e, '--o', ctrl_t, ctrl_e, '--o',\
                 sobol_t, sobol_e, '--o', fft_t, fft_e, '--o')
        # plt.legend(['vanilla MC', 'antithetic variates', 'control variates', 'sobol seq', 'sobol seq + ctrl var', 'FFT Conv'])
        plt.legend(['vanilla MC', 'control variates', 'sobol seq', 'FFT Conv'])
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('second')
        plt.ylabel('Relative error')
        # plt.title('Standard Deviation of Different Variation Reduction Techniques')
        # plt.savefig("std.png", dpi=1000)
        plt.show()

if __name__ == '__main__':
    unittest.main()