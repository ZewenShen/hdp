import numpy as np
from scipy import interpolate

class Characteristic:
    """
    u_t + uu_x = 0
    """
    def __init__(self, domain, du0):
        """
        du0: the first derivative of initial condition u0
        """
        assert domain.ic is not None
        self.domain = domain
        self.u0_func = domain.ic
        self.du0_func = du0
    
    def solve(self, nx, nt):
        a, b, T = self.domain.a, self.domain.b, self.domain.T
        X = np.linspace(a, b, nx+1)
        tmp_T = -1/np.min(self.du0_func(X, 0))
        if tmp_T > 0: T = tmp_T
        ht = T / nt; self.ht = ht
        self.max_T = T - ht if tmp_T > 0 else T
        
        u0 = self.u0_func(X, 0); self.u0 = u0
        self.time_vec = np.linspace(0, T, nt+1)
        self.characteristic_lines = X.reshape((1, -1))
        for i in range(1, nt+1):
            self.characteristic_lines = np.vstack([self.characteristic_lines, X + i*ht*u0])

    def evaluate(self, X, t, interp_method='cubic'):
        t_index = int(round(t/self.ht))
        domain = self.characteristic_lines[t_index]
        f = interpolate.interp1d(domain, self.u0, 'cubic')
        return f(X)
        

