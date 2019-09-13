class Domain1d:

    def __init__(self, a, b, T, ic=None, bc=None):
        self.a, self.b = a, b
        self.T = T
        self.ic, self.bc = ic, bc
    
    def get_discretization_size(self, nx, nt):
        hx, ht = (self.b-self.a)/nx, self.T/nt
        return hx, ht

class Domain2d(Domain1d):

    def __init__(self, a, b, c, d, T, ic=None, bc=None):
        super().__init__(a, b, T, ic, bc)
        self.c, self.d = c, d
    
    def get_discretization_size(self, nx, ny, nt):
        hx, hy, ht = (self.b-self.a)/nx, (self.d-self.c)/ny, self.T/nt
        return hx, hy, ht