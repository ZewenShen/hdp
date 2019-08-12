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
        last_price = list(map(lambda x: x[:, -1], self.simulation_result))
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
    
