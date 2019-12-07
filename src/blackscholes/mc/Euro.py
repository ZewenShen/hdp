import numpy as np
from math import sqrt
from scipy.stats import norm

class Euro:
    """
    Vanilla Multi-Dimensional European Option
    """
    def __init__(self, payoff_func, random_walk):
        """
        payoff: A function that takes ${asset_num} variables as input, returns the a scalar payoff
        random_walk: A random walk generator, e.g. GBM (geometric brownian motion)
        """
        self.payoff_func = payoff_func
        self.random_walk = random_walk

    def price(self, path_num=1000, ci=False):
        self.simulation_result = self.random_walk.simulate(path_num)
        last_price = [x[:, -1] for x in self.simulation_result]
        payoff = self.payoff_func(last_price) * np.exp(-self.random_walk.ir * self.random_walk.T)
        value = np.mean(payoff) 
        if ci:
            std = np.std(payoff, ddof=1)
            return value, std
        else:
            return value

    def priceV2(self, path_num=10000, ci=False):
        """
        Stock prices approximated by the analytical solution to the SDE.
        """
        self.simulation_result = self.random_walk.simulateV2_T(path_num)
        payoff = self.payoff_func(self.simulation_result) * np.exp(-self.random_walk.ir * self.random_walk.T)
        value = np.mean(payoff) 
        if ci:
            std = np.std(payoff, ddof=1)
            return value, std
        else:
            return value

    def priceV3(self, path_num=10000, ci=False):
        """
        Stock prices approximated by the analytical solution to the SDE, but solutions at each time steps are all given.
        """
        self.simulation_result = self.random_walk.simulateV2(path_num)
        last_price = np.array([x[:, -1] for x in self.simulation_result])
        payoff = self.payoff_func(last_price) * np.exp(-self.random_walk.ir * self.random_walk.T)
        value = np.mean(payoff) 
        if ci:
            std = np.std(payoff, ddof=1)
            return value, std
        else:
            return value

    def priceV4(self, path_num=10000, ci=False):
        """
        Stock prices approximated by the analytical solution to the SDE. SDE simulated by the sobol sequence.
        """
        self.simulation_result = self.random_walk.simulateV4_T(path_num)
        payoff = self.payoff_func(self.simulation_result) * np.exp(-self.random_walk.ir * self.random_walk.T)
        value = np.mean(payoff) 
        if ci:
            std = np.std(payoff, ddof=1)
            return value, std
        else:
            return value
    
    def priceV5(self, path_num=10000, ci=False):
        """
        Stock prices approximated by the analytical solution to the SDE with antithetic variates.
        """
        self.simulation_result = self.random_walk.simulateV2_T_antithetic(int(path_num/2))
        payoff = self.payoff_func(self.simulation_result) * np.exp(-self.random_walk.ir * self.random_walk.T)
        value = np.mean(payoff) 
        if ci:
            std = np.std(payoff, ddof=1)
            return value, std
        else:
            return value

    def priceV6(self, path_num=10000, ci=False):
        """
        Stock prices approximated by the analytical solution to the SDE with antithetic variates and sobol seq.
        """
        self.simulation_result = self.random_walk.simulateV4_T_antithetic(int(path_num/2))
        payoff = self.payoff_func(self.simulation_result) * np.exp(-self.random_walk.ir * self.random_walk.T)
        value = np.mean(payoff)
        if ci:
            std = np.std(payoff, ddof=1)
            return value, std
        else:
            return value
        
    def priceV7(self, path_num=10000):
        """
        Stock prices approximated by the analytical solution to the SDE with control variates.
        """
        self.simulation_result = self.random_walk.simulateV2_T(path_num)
        expectation = np.exp((self.random_walk.ir-self.random_walk.dividend_vec)*self.random_walk.T)\
            * self.random_walk.init_price_vec
        payoff = self.payoff_func(self.simulation_result) * np.exp(-self.random_walk.ir * self.random_walk.T)
        cov = np.cov(self.simulation_result.T, payoff)
        Sx = cov[:-1, :-1]
        Sxy = cov[-1, :-1]
        b_hat = np.linalg.inv(Sx) @ Sxy
        return np.mean(payoff) - b_hat.dot(np.mean(self.simulation_result, axis=0) - expectation)

    def priceV8(self, path_num=10000):
        """
        Stock prices approximated by the analytical solution to the SDE with control variates and sobol seq.
        """
        self.simulation_result = self.random_walk.simulateV4_T(path_num)
        expectation = np.exp((self.random_walk.ir-self.random_walk.dividend_vec)*self.random_walk.T)\
            * self.random_walk.init_price_vec
        payoff = self.payoff_func(self.simulation_result) * np.exp(-self.random_walk.ir * self.random_walk.T)
        cov = np.cov(self.simulation_result.T, payoff)
        Sx = cov[:-1, :-1]
        Sxy = cov[-1, :-1]
        b_hat = np.linalg.inv(Sx) @ Sxy
        return np.mean(payoff) - b_hat.dot(np.mean(self.simulation_result, axis=0) - expectation)


    def price1d_control_variates(self, path_num=1000):
        assert len(self.random_walk.init_price_vec) == 1
        self.simulation_result = self.random_walk.simulate(path_num)
        last_price = [x[:, -1] for x in self.simulation_result]
        X = np.array(last_price).ravel()
        Y = self.payoff_func(last_price) * np.exp(-self.random_walk.ir*self.random_walk.T)
        meanX, meanY = np.mean(X), np.mean(Y)
        b_hat = np.sum(np.multiply(X-meanX, Y-meanY))/np.sum(np.power(X-meanX, 2))
        return np.mean(Y-b_hat*(last_price - np.exp(self.random_walk.ir*self.random_walk.T)*self.random_walk.init_price_vec[0]))

    def price_antithetic_variates(self, path_num=1000):
        self.simulation_result = self.random_walk.antithetic_simulate(path_num)
        last_price = [x[:, -1] for x in self.simulation_result]
        payoff = self.payoff_func(last_price)
        return np.mean(payoff) * np.exp(-self.random_walk.ir * self.random_walk.T)

    def price_importance_sampling(self, path_num):
        assert hasattr(self.payoff_func, 'strike'), 'Meta info of the payoff function lost: strike'
        strike = self.payoff_func.strike
        self.simulation_result, Zs = self.random_walk.importance_sampling_simulate_T(path_num, strike)
        drift_vec = self.random_walk.ir - self.random_walk.dividend_vec
        norm_mean_old = (drift_vec - self.random_walk.vol_vec**2/2)*self.random_walk.T
        norm_mean_new = np.log(strike/self.random_walk.init_price_vec) - self.random_walk.T*self.random_walk.vol_vec**2/2
        scale = self.random_walk.vol_vec * sqrt(self.random_walk.T)
        density_ratio = norm.pdf(Zs, loc=norm_mean_old, scale=scale)/norm.pdf(Zs, loc=norm_mean_new, scale=scale)
        payoff = self.payoff_func(self.simulation_result)
        return np.mean(np.multiply(payoff, density_ratio.T)) * np.exp(-self.random_walk.ir*self.random_walk.T)


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../../')
    from blackscholes.utils.GBM import GBM

    init_price_vec = np.ones(5)
    vol_vec = 0.2*np.ones(5)
    ir = 0.00
    dividend_vec = np.zeros(5)
    corr_mat = np.eye(5)
    random_walk = GBM(3, 100, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
    def test_payoff(*l):
        return max(np.sum(l) - 5, 0)
    a = Euro(test_payoff, random_walk).price(10000)
    print(a)
    
