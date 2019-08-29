import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy import interpolate


class Domain1d:

    def __init__(self, a, b, T, ic=None, bc=None):
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
        self.nx = nx
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
            B1 = np.multiply(self.p(X, prev_t)/(2*hx**2) - self.q(X, prev_t)/(4*hx), solution[-1][:-2])
            B2 = np.multiply(1/ht - self.p(X, prev_t)/hx**2 + self.r(X, prev_t)/2, solution[-1][1:-1])
            B3 = np.multiply(self.p(X, prev_t)/(2*hx**2) + self.q(X, prev_t)/(4*hx), solution[-1][2:])
            B += B1 + B2 + B3 + (self.f(X, t) + self.f(X, prev_t))/2
            x = spsolve(A.tocsr(), B)
            x = np.concatenate([[self.domain.bc(self.domain.a, t)], x, [self.domain.bc(self.domain.b, t)]])
            solution = np.vstack([solution, x])
        self.solution = solution
        return solution
    
    def evaluate(self, points):
        domain = np.linspace(self.domain.a, self.domain.b, self.nx+1)
        f = interpolate.interp1d(domain, self.solution[-1], 'cubic')
        return f(points)