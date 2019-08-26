import numpy as np

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

    def price(self, path_num=1000):
        self.simulation_result = self.random_walk.simulate(path_num)
        last_price = [x[:, -1] for x in self.simulation_result]
        payoff = list(map(self.payoff_func, last_price))
        return np.mean(payoff) * np.exp(-self.random_walk.ir * self.random_walk.T)

    def priceV2(self, path_num=10000):
        """
        Stock prices approximated by the analytical solution to the SDE.
        """
        self.simulation_result = self.random_walk.simulateV2_T(path_num)
        payoff = list(map(self.payoff_func, self.simulation_result))
        return np.mean(payoff) * np.exp(-self.random_walk.ir * self.random_walk.T)

    def priceV3(self, path_num=10000):
        """
        Stock prices approximated by the analytical solution to the SDE, but solutions at each time steps are all given.
        """
        self.simulation_result = self.random_walk.simulateV2(path_num)
        last_price = [x[:, -1] for x in self.simulation_result]
        payoff = list(map(self.payoff_func, last_price))
        return np.mean(payoff) * np.exp(-self.random_walk.ir * self.random_walk.T)
    
    def price1d_control_variates(self, path_num=1000):
        assert len(self.random_walk.init_price_vec) == 1
        self.simulation_result = self.random_walk.simulate(path_num)
        last_price = [x[:, -1] for x in self.simulation_result]
        X = np.array(last_price).ravel()
        Y = np.array([self.payoff_func(p) for p in last_price])*np.exp(-self.random_walk.ir*self.random_walk.T)
        meanX, meanY = np.mean(X), np.mean(Y)
        b_hat = np.sum(np.multiply(X-meanX, Y-meanY))/np.sum(np.power(X-meanX, 2))
        return np.mean(Y-b_hat*(last_price - np.exp(self.random_walk.ir*self.random_walk.T)*self.random_walk.init_price_vec[0]))

    def price_antithetic_variates(self, path_num=1000):
        self.simulation_result = self.random_walk.antithetic_simulate(path_num)
        last_price = [x[:, -1] for x in self.simulation_result]
        payoff = list(map(self.payoff_func, last_price))
        return np.mean(payoff) * np.exp(-self.random_walk.ir * self.random_walk.T)

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
    
