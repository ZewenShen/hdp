import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../..")
from blackscholes.utils.Regression import Regression
import numpy as np

class American:
    """
    Multi-Dimensional American Option. Priced by the Least Square Monte Carlo method.
    """
    def __init__(self, payoff_func, random_walk):
        """
        payoff: A function that takes ${asset_num} variables as input, returns the a scalar payoff
        random_walk: A random walk generator, e.g. GBM (geometric brownian motion)
        """
        self.payoff_func = payoff_func
        self.random_walk = random_walk
    
    def price(self, path_num=1000):
        """Least Square Monte Carlo method"""
        self.simulation_result = self.random_walk.simulateV2(path_num)
        cashflow_matrix = np.zeros([path_num, self.random_walk.N+1])
        cur_price = np.array([x[:, -1] for x in self.simulation_result])
        cur_payoff = np.array(list(map(self.payoff_func, cur_price)))
        cashflow_matrix[:, self.random_walk.N] = cur_payoff
        for t in range(self.random_walk.N-1, 0, -1):
            discounted_cashflow = self._get_discounted_cashflow(t, cashflow_matrix, path_num)
            # Compute the discounted payoff
            r = Regression(self.simulation_result[:, :, t], discounted_cashflow, payoff_func=self.payoff_func)
            if not r.has_intrinsic_value: continue # Intrinsic value = 0
            cur_price = np.array([x[:, t] for x in self.simulation_result])
            cur_payoff = np.array(list(map(self.payoff_func, cur_price[r.index])))
            continuation = np.array([r.evaluate(X) for X in cur_price[r.index]])
            exercise_index = r.index[cur_payoff >= continuation]
            cashflow_matrix[exercise_index] = np.zeros(cashflow_matrix[exercise_index].shape)
            cashflow_matrix[exercise_index, t] = np.array(list(map(self.payoff_func, cur_price)))[exercise_index]
        return self._get_discounted_cashflow_at_t0(cashflow_matrix)

    def _get_discounted_cashflow(self, t, cashflow_matrix, path_num):
        discounted_cashflow = np.zeros(path_num)
        for i in range(len(cashflow_matrix)):
            cashflow = cashflow_matrix[i]
            for j in range(self.random_walk.N, t, -1):
                if cashflow[j] != 0:
                    discounted_cashflow[i] = cashflow[j] * np.exp((t-j)*self.random_walk.dt*self.random_walk.ir)
                    break
        return discounted_cashflow

    def _get_discounted_cashflow_at_t0(self, cashflow_matrix):
        summation = 0
        for cashflow in cashflow_matrix:
            for i in range(1, len(cashflow)):
                if cashflow[i] != 0:
                    summation += cashflow[i]*np.exp(-self.random_walk.ir*i*self.random_walk.dt)
                    break
        return summation / len(cashflow_matrix)