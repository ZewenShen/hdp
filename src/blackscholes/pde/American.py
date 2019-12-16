import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../..")
from blackscholes.pde.Parabolic import Solver1d_penalty
import numpy as np

class Amer1d(Solver1d_penalty):
    
    def __init__(self, domain, vol, ir, dividend, strike, cp_type):
        """
        cp_type (call/put type): 1 if call, -1 if put
        """
        p = lambda S, t: vol**2*S**2/2
        q = lambda S, t: (ir-dividend)*S
        r = lambda S, t: -ir*np.ones(len(S))
        f = lambda S, t: 0
        domain.ic = lambda S, t: np.maximum(cp_type*(S - strike), 0)
        domain.bc = lambda S, t: strike*np.exp(-ir*t) if abs(S) < 7/3-4/3-1 else 0
        super().__init__(p, q, r, f, domain)
