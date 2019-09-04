import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags, kron
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
    u_t = p(x, t)u_xx + q(x, t)u_x + r(x, t)u + f(x, t)
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

class Domain2d(Domain1d):

    def __init__(self, a, b, c, d, T, ic=None, bc=None):
        super().__init__(a, b, T, ic, bc)
        self.c, self.d = c, d
    
    def get_discretization_size(self, nx, ny, nt):
        hx, hy, ht = (self.b-self.a)/nx, (self.d-self.c)/ny, self.T/nt
        return hx, hy, ht

class Coef2d:
    """
    Represent a coefficient of differential operator \sum_{i=0}^n f1(x, t)f2(y, t)
    """
    def __init__(self, f1, f2):
        """
        f1: f1(x, t)
        f2: f2(y, t)
        They are either functions or functions in a list (python list or numpy list)
        """
        if callable(f1) and callable(f2):
            self.f1 = [f1]
            self.f2 = [f2]
        elif len(f1) == len(f2):
            self.f1 = f1
            self.f2 = f2
        else:
            raise RuntimeError("2D solver input coefficient error")

class Solver2d:
    """
    u_t = coef_a*u_xx + coef_b*u_xy + coef_c*u_yy + coef_d*u_x + coef_e*u_y + coef_f*u + g(x, y, t)
    """
    def __init__(self, coef_a, coef_b, coef_c, coef_d, coef_e, coef_f, g, domain):
        self.coef_a, self.coef_b, self.coef_c, self.coef_d, self.coef_e, self.coef_f, self.g = coef_a, coef_b, coef_c, coef_d, coef_e, coef_f, g
        self.domain = domain
    
    def solve(self, nx, ny, nt):
        s1, s2 = np.linspace(self.domain.a, self.domain.b, nx+1), np.linspace(self.domain.c, self.domain.d, ny+1)
        hx, hy, ht = self.domain.get_discretization_size(nx, ny, nt)
        Is1, Is2 = np.eye(nx), np.eye(ny)
        T2s1 = diags([1, -2, 1], [-1, 0, 1], (nx-1, nx-1))/hx**2
        T1s1 = diags([-1, 0, 1], [-1, 0, 1], (nx-1, nx-1))/(2*hx)
        T2s2 = diags([1, -2, 1], [-1, 0, 1], (ny-1, ny-1))/hy**2
        T1s2 = diags([-1, 0, 1], [-1, 0, 1], (ny-1, ny-1))/(2*hy)
        for i in range(1, nt+1):
            t, prev_t = i*ht, (i-1)*ht
            """
            A20, C20 = kron(T2s1, Is2), kron(diags(self.a(s1, s2, t)), Is2)
            A02, C02 = kron(Is1, T2s2), kron(Is1, diags(self.b(s1, s2, t)))
            A11, C11 = kron(T1s1, T1s2), kron
            A10, C10 = kron(T1s1, Is2), kron(diags(self.d(s1, s2, t)), Is2)
            A01, C01 = kron(Is1, T1s2), kron(Is1, diags(self.e(s1, s2, t)))
            A00, C00 = kron(Is1, Is2), self.f(s1, s2, t)*kron(Is1, Is2)
            """
