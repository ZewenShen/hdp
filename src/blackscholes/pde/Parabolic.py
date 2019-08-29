import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

class Domain1d:

    def __init__(self, a, b, T, ic, bc):
        self.a, self.b = a, b
        self.T = T
        self.ic, self.bc = ic, bc
    
    def get_discretization_size(self, nx, nt):
        hx, ht = (self.b-self.a)/nx, self.T/nt
        return hx, ht

class Solver1d:
    """
    u_t = p(x, t)u_xx + q(x, t)u_x + r(x, t)u + f
    """
    def __init__(self, p, q, r, f, domain):
        self.p, self.q, self.r, self.f = p, q, r, f
        self.domain = domain

    def solve(self, nx, nt):
        hx, ht = self.domain.get_discretization_size(nx, nt)
        domain = np.linspace(self.domain.a, self.domain.b, nx+1)
        X = domain[1:-1]
        solution = self.domain.ic(domain, 0)
        solution = np.reshape(solution, (1, -1))
        for i in range(1, nt+1):
            t, prev_t = i*ht, (i-1)*ht
            A = lil_matrix((nx-1, nx-1))
            B = np.zeros(nx-1)
            A.setdiag(-self.p(X[1:], t)/(2*hx**2) + self.q(X[1:], t)/(4*hx), k=-1)
            A.setdiag(1/ht + self.p(X, t)/hx**2 - self.r(X, t)/2, k=0)
            A.setdiag(-self.p(X[:-1], t)/(2*hx**2) - self.q(X[:-1], t)/(4*hx), k=1)
            B[0] -= (-self.p(X[1], t)/(2*hx**2) + self.q(X[1], t)/(4*hx))*self.domain.bc(self.domain.a, t)
            B[-1] -= (-self.p(X[-1], t)/(2*hx**2) - self.q(X[-1], t)/(4*hx))*self.domain.bc(self.domain.b, t)
