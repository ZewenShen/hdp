import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
import numpy as np
from utils.Domain import Domain1d

class Characteristic:
    """
    u_t + uu_x = 0
    """
    def __init__(self, Domain1d, du0):
        """
        du0: the first derivative of initial condition u0
        """
        assert Domain1d.ic is not None
        self.domain = Domain1d
        self.u0_func = Domain1d.ic
        self.du0_func = du0
    
    def solve(self, nx, nt):
        a, b, T = self.domain.a, self.domain.b, self.domain.T
        X = np.linspace(a, b, nx+1)
        T = np.max(np.min(-1/self.du0_func(X)), T)
        u0 = self.u0_func(X, 0)
        self.time_vec = np.linspace(0, T, nt+1)
        # solution = self.u0_func(X, 0)
        # solution = np.reshape(solution, (1, -1))

