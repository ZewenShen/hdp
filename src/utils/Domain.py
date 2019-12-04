import numpy as np
from functools import reduce

class Domain1d:

    def __init__(self, a, b, T, ic=None, bc=None):
        self.a, self.b = a, b
        self.T = T
        self.ic, self.bc = ic, bc
    
    def get_discretization_size(self, nx, nt):
        hx, ht = (self.b-self.a)/nx, self.T/nt
        return hx, ht

class Sampler1d:

    def __init__(self, domain1d, S_multiplier=1.2):
        """
        S_multiplier: Sample from a domain larger than the preset domain
        """
        self.domain = domain1d
        self.S_multiplier = S_multiplier
    
    def run(self, n_interior, n_boundary, n_terminal):
        # Sampler #1: domain interior
        a, b, T, S_multiplier = self.domain.a, self.domain.b, self.domain.T, self.S_multiplier
        t_interior = np.random.uniform(low=0, high=T-1e-10, size=[n_interior, 1])
        S_interior = np.random.uniform(low=a/S_multiplier, high=b*S_multiplier, size=[n_interior, 1])

        # Sampler #2: spatial boundary
        t_boundary = np.random.uniform(low=0, high=T-1e-10, size=[n_boundary, 1])
        S_boundary = np.random.choice([a/S_multiplier, b*S_multiplier], n_boundary).reshape(n_boundary, 1)
        
        # Sampler #3: initial/terminal condition
        t_terminal = T * np.ones((n_terminal, 1))
        S_terminal = np.random.uniform(low=a/S_multiplier, high=b*S_multiplier, size = [n_terminal, 1])
        
        return S_interior, t_interior, S_boundary, t_boundary, S_terminal, t_terminal

class Sampler1dBoundary2Center:

    def __init__(self, domain1d, S_multiplier=1.2):
        """
        S_multiplier: Sample from a domain larger than the preset domain
        """
        self.domain = domain1d
        self.S_multiplier = S_multiplier
        self.i = 0
    
    def run(self, n_interior, n_boundary, n_terminal):
        # Sampler #1: domain interior
        a, b, T, S_multiplier = self.domain.a, self.domain.b, self.domain.T, self.S_multiplier

        i = self.i
        t_interior = np.random.uniform(low=0, high=T-1e-10, size=[min(int(2*i)+n_boundary+n_terminal, n_interior), 1])
        S_interior = np.random.uniform(low=a/S_multiplier, high=b*S_multiplier, size=[min(int(2*i)+n_boundary+n_terminal, n_interior), 1])

        # Sampler #2: spatial boundary
        t_boundary = np.random.uniform(low=0, high=T-1e-10, size=[max(int(n_interior/2-i), n_boundary), 1])
        S_boundary = np.random.choice([a/S_multiplier, b*S_multiplier], max(int(n_interior/2-i), n_boundary)).reshape(max(int(n_interior/2-i), n_boundary), 1)
        
        # Sampler #3: initial/terminal condition
        t_terminal = T * np.ones((max(int(n_interior/2-i), n_terminal), 1))
        S_terminal = np.random.uniform(low=a/S_multiplier, high=b*S_multiplier, size = [max(int(n_interior/2-i), n_terminal), 1])
        
        self.i += 0.25

        return S_interior, t_interior, S_boundary, t_boundary, S_terminal, t_terminal

class Domain2d(Domain1d):

    def __init__(self, a, b, c, d, T, ic=None, bc=None):
        super().__init__(a, b, T, ic, bc)
        self.c, self.d = c, d
    
    def get_discretization_size(self, nx, ny, nt):
        hx, hy, ht = (self.b-self.a)/nx, (self.d-self.c)/ny, self.T/nt
        return hx, hy, ht

class SamplerNd:
    
    def __init__(self, domainNd):
        self.domain = domainNd
    
    def run(self, n_interior, n_terminal):
        # Sampler #1: domain interior
        dim, T = self.domain.dim, self.domain.T
        lower_bound, upper_bound = self.domain.lower_bound, self.domain.upper_bound
        t_interior = np.random.uniform(low=0, high=T-1e-10, size=[n_interior, 1])
        S_interior = np.random.uniform(low=lower_bound, high=upper_bound, size=[n_interior, dim])

        # Sampler #2: spatial boundary
        # t_boundary = np.random.uniform(low=0, high=T-1e-10, size=[n_boundary, 1])
        # S_boundary = np.random.choice([a/S_multiplier, b*S_multiplier], n_boundary).reshape(n_boundary, 1)
        
        # Sampler #3: initial/terminal condition
        t_terminal = T * np.ones((n_terminal, 1))
        S_terminal = np.random.uniform(low=lower_bound, high=upper_bound, size=[n_terminal, dim])
        
        return S_interior, t_interior, S_terminal, t_terminal

class SamplerNdV2:
    
    def __init__(self, domainNd):
        self.domain = domainNd
    
    def run(self, n_interior, n_boundary, n_terminal):
        # Sampler #1: domain interior
        dim, T = self.domain.dim, self.domain.T
        lower_bound, upper_bound = self.domain.lower_bound, self.domain.upper_bound
        t_interior = np.random.uniform(low=0, high=T-1e-10, size=[n_interior, 1])
        S_interior = np.random.uniform(low=lower_bound, high=upper_bound, size=[n_interior, dim])

        # Sampler #2: spatial boundary
        tmin_boundarys = []
        Smin_boundarys = []
        for i in range(dim):
            t_boundary = np.random.uniform(low=0, high=T-1e-10, size=[n_boundary//2//dim, 1])
            S_boundary = np.random.uniform(low=lower_bound, high=upper_bound, size=[n_boundary//2//dim, dim])
            S_boundary[:, i] = lower_bound[i]
            tmin_boundarys.append(t_boundary)
            Smin_boundarys.append(S_boundary)
        tmin_boundarys = reduce(lambda x, y: np.vstack((x, y)), tmin_boundarys)
        Smin_boundarys = reduce(lambda x, y: np.vstack((x, y)), Smin_boundarys)

        tmax_boundarys = []
        Smax_boundarys = []
        for i in range(dim):
            t_boundary = np.random.uniform(low=0, high=T-1e-10, size=[n_boundary//2//dim, 1])
            S_boundary = np.random.uniform(low=lower_bound, high=upper_bound, size=[n_boundary//2//dim, dim])
            S_boundary[:, i] = upper_bound[i]
            tmax_boundarys.append(t_boundary)
            Smax_boundarys.append(S_boundary)
        tmax_boundarys = reduce(lambda x, y: np.vstack((x, y)), tmax_boundarys)
        Smax_boundarys = reduce(lambda x, y: np.vstack((x, y)), Smax_boundarys)

        # Sampler #3: initial/terminal condition
        t_terminal = T * np.ones((n_terminal, 1))
        S_terminal = np.random.uniform(low=lower_bound, high=upper_bound, size=[n_terminal, dim])
        
        return S_interior, t_interior, Smin_boundarys, tmin_boundarys, Smax_boundarys, tmax_boundarys, S_terminal, t_terminal

class DomainNd:

    def __init__(self, range_vec, T):
        self.dim = len(range_vec)
        self.range_vec = range_vec
        self.lower_bound = range_vec.T[0]
        self.upper_bound = range_vec.T[1]
        self.T = T
    
    def get_discretization_size(self, n_vec, nt):
        return (self.range_vec[:, 1] - self.range_vec[:, 0]) / n_vec, self.T / nt