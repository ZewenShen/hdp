import numpy as np
class GBM:
    """
    Multi-asset geometric brownian motion simulation
    """
    def __init__(self, T, N, init_price_vec, ir_vec, vol_vec, dividend_vec, corr_mat):
        assert len(init_price_vec) == len(ir_vec) == len(vol_vec) == len(dividend_vec) == corr_mat.shape[0] == corr_mat.shape[1], "Vectors' lengths are different"
        self.dt = T/N
        self.N = N
        self.init_price_vec = init_price_vec
        self.ir_vec = ir_vec
        self.vol_vec = vol_vec
        self.dividend_vec = dividend_vec
        self.corr_mat = corr_mat
        self.asset_num = len(init_price_vec)
    
    def simulate(self, M):
        """
        M: total simulation number
        """
        simulations = []
        drift_vec = self.ir_vec - self.dividend_vec
        L = np.linalg.cholesky(self.corr_mat)
        for _ in range(M):
            sim = np.zeros([self.asset_num, self.N+1])
            sim[:, 0] = self.init_price_vec
            for i in range(1, self.N+1):
                dW = np.zeros(self.asset_num)
                for k in range(self.asset_num):
                    normal = np.random.normal(size=k+1)
                    for j in range(k+1):
                        dW[k] += L[k, j]*normal[j]
                    dW[k] *= self.dt**0.5
                sim[:, i] = np.multiply(1 + self.dt*drift_vec, sim[:, i-1]) + np.multiply(np.multiply(self.vol_vec, sim[:, i-1]), dW)
            simulations.append(sim)
        return simulations if len(simulations) > 1 else simulations[0]



if __name__ == "__main__":
    init_price_vec = np.ones(5)
    vol_vec = 0.2*np.ones(5)
    ir_vec = 0.00*np.ones(5)
    dividend_vec = np.zeros(5)
    corr_mat = np.eye(5)
    a = GBM(3, 100, init_price_vec, ir_vec, vol_vec, dividend_vec, corr_mat).simulate(1)
    print(a)