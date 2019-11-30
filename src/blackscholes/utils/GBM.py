import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../..")
from utils.Sobol import i4_sobol_generate_std_normal
import numpy as np
from math import sqrt
import random

class GBM:
    """
    Multi-asset geometric brownian motion simulation
    """
    def __init__(self, T, N, init_price_vec, ir, vol_vec, dividend_vec, corr_mat):
        assert len(init_price_vec) == len(vol_vec) == len(dividend_vec) == corr_mat.shape[0] == corr_mat.shape[1], "Vectors' lengths are different"
        self.dt = T/N
        self.T = T
        self.N = N
        self.init_price_vec = init_price_vec
        self.ir = ir
        self.vol_vec = vol_vec
        self.dividend_vec = dividend_vec
        self.corr_mat = corr_mat
        self.asset_num = len(init_price_vec)
    
    def simulate(self, M):
        """
        M: total simulation number
        """
        simulations = []
        drift_vec = self.ir - self.dividend_vec
        L = np.linalg.cholesky(self.corr_mat)
        for _ in range(M):
            sim = np.zeros([self.asset_num, self.N+1])
            sim[:, 0] = self.init_price_vec
            for i in range(1, self.N+1):
                dW = L.dot(np.random.normal(size=self.asset_num))*sqrt(self.dt)
                sim[:, i] = np.multiply(1 + self.dt*drift_vec, sim[:, i-1]) + np.multiply(np.multiply(self.vol_vec, sim[:, i-1]), dW)
            simulations.append(sim)
        return np.array(simulations)
    
    def simulateV2(self, M):
        """
        Vanilla simulation using the lognormal distribution.
        """
        simulations = []
        drift_vec = self.ir - self.dividend_vec
        L = np.linalg.cholesky(self.corr_mat)
        for _ in range(M):
            sim = np.zeros([self.asset_num, self.N+1])
            sim[:, 0] = self.init_price_vec
            for i in range(1, self.N+1):
                dW = L.dot(np.random.normal(size=self.asset_num))*sqrt(self.dt)
                rand_term = np.multiply(self.vol_vec, dW)
                sim[:, i] = np.multiply(sim[:, i-1], np.exp((drift_vec-self.vol_vec**2/2)*self.dt + rand_term))
            simulations.append(sim)
        return np.array(simulations)

    def simulateV2_T(self, M):
        """
        Vanilla simulation using the lognormal distribution.
        Only returns the stock price at time T.
        """
        simulations = []
        drift_vec = self.ir - self.dividend_vec
        fixed = np.multiply(self.init_price_vec, np.exp((drift_vec-self.vol_vec**2/2)*self.T))
        L = np.linalg.cholesky(self.corr_mat)
        for _ in range(M):
            dW = L.dot(np.random.normal(size=self.asset_num))*sqrt(self.T)
            rand_term = np.multiply(self.vol_vec, dW)
            sim = np.multiply(fixed, np.exp(rand_term))
            simulations.append(sim)
        return np.array(simulations)
    
    def simulateV4_T(self, M):
        """
        Vanilla simulation using the lognormal distribution.
        Only returns the stock price at time T.
        RNG: the sobol sequence
        """
        simulations = []
        drift_vec = self.ir - self.dividend_vec
        fixed = np.multiply(self.init_price_vec, np.exp((drift_vec-self.vol_vec**2/2)*self.T))
        L = np.linalg.cholesky(self.corr_mat)
        sobol_seq = i4_sobol_generate_std_normal(self.asset_num, M)
        for i in range(M):
            dW = L.dot(sobol_seq[i])*sqrt(self.T)
            rand_term = np.multiply(self.vol_vec, dW)
            sim = np.multiply(fixed, np.exp(rand_term))
            simulations.append(sim)
        return np.array(simulations)
    
    def simulateV2_T_antithetic(self, M):
        """
        Vanilla simulation using the lognormal distribution with antithetic variates.
        Only returns the stock price at time T.
        """
        simulations = []
        drift_vec = self.ir - self.dividend_vec
        fixed = np.multiply(self.init_price_vec, np.exp((drift_vec-self.vol_vec**2/2)*self.T))
        L = np.linalg.cholesky(self.corr_mat)
        for _ in range(M):
            dW = L.dot(np.random.normal(size=self.asset_num))*sqrt(self.T)
            rand_term = np.multiply(self.vol_vec, dW)
            sim = np.multiply(fixed, np.exp(rand_term))
            sim_antithetic = np.multiply(fixed, np.exp(-rand_term))
            simulations.append(sim)
            simulations.append(sim_antithetic)
        return np.array(simulations)

    def simulateV4_T_antithetic(self, M):
        """
        Vanilla simulation using the lognormal distribution.
        Only returns the stock price at time T.
        RNG: the sobol sequence
        """
        simulations = []
        drift_vec = self.ir - self.dividend_vec
        fixed = np.multiply(self.init_price_vec, np.exp((drift_vec-self.vol_vec**2/2)*self.T))
        L = np.linalg.cholesky(self.corr_mat)
        sobol_seq = i4_sobol_generate_std_normal(self.asset_num, M)
        for i in range(M):
            dW = L.dot(sobol_seq[i])*sqrt(self.T)
            rand_term = np.multiply(self.vol_vec, dW)
            sim = np.multiply(fixed, np.exp(rand_term))
            sim_antithetic = np.multiply(fixed, np.exp(-rand_term))
            simulations.append(sim)
            simulations.append(sim_antithetic)
        return np.array(simulations)

    def antithetic_simulate(self, M):
        simulations = []
        drift_vec = self.ir - self.dividend_vec
        L = np.linalg.cholesky(self.corr_mat)
        for _ in range(M//2):
            sim, antithetic_sim = np.zeros([self.asset_num, self.N+1]), np.zeros([self.asset_num, self.N+1])
            sim[:, 0], antithetic_sim[:, 0] = self.init_price_vec, self.init_price_vec
            for i in range(1, self.N+1):
                dW = L.dot(np.random.normal(size=self.asset_num))*sqrt(self.dt)
                antithetic_dW = -dW
                sim[:, i] = np.multiply(1 + self.dt*drift_vec, sim[:, i-1]) +\
                            np.multiply(np.multiply(self.vol_vec, sim[:, i-1]), dW)
                antithetic_sim[:, i] = np.multiply(1 + self.dt*drift_vec, antithetic_sim[:, i-1]) +\
                                       np.multiply(np.multiply(self.vol_vec, antithetic_sim[:, i-1]), antithetic_dW)
            simulations.append(sim)
            simulations.append(antithetic_sim)
        return np.array(simulations)
    
    def importance_sampling_simulate_T(self, M, strike):
        """
        Currently only support Hockey stick payoff functions.
        """
        simulations = []
        Zs = []
        L = np.linalg.cholesky(self.corr_mat)
        for _ in range(M):
            dW = L.dot(np.random.normal(size=self.asset_num))*sqrt(self.T)
            rand_term = np.multiply(self.vol_vec, dW)
            Z = np.log(strike/self.init_price_vec) - self.T*self.vol_vec**2/2 + rand_term
            sim = np.multiply(self.init_price_vec, np.exp(Z))
            Zs.append(Z)
            simulations.append(sim)
        return np.array(simulations), np.array(Zs)

if __name__ == "__main__":
    init_price_vec = np.ones(5)
    vol_vec = 0.2*np.ones(5)
    ir = 0
    dividend_vec = np.zeros(5)
    corr_mat = np.eye(5)
    a = GBM(3, 100, init_price_vec, ir, vol_vec, dividend_vec, corr_mat).simulate(1)
    print(a)