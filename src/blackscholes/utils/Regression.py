import numpy as np

class Monomial:

    def __init__(self, a_vec):
        self.a_vec = np.array(a_vec)
        self.dimension = len(a_vec)
    
    def evaluate(self, input_vec):
        assert self.dimension == len(input_vec), "Input data dimension is different from the monomial's dimension"
        return np.prod(np.power(input_vec, self.a_vec))

class Monomial_Basis:

    def __init__(self, chi, dimension):
        permutations = Monomial_Basis._get_all_permutations(chi, dimension)
        self.monomials = [Monomial(x) for x in permutations]
    
    @staticmethod
    def _get_all_permutations(chi, dimension):
        if chi == 0:
            return [[0]*dimension]
        elif dimension == 1:
            return [[i] for i in range(chi+1)]
        else:
            results = []
            for i in range(chi+1):
                results += [[i] + x for x in Monomial_Basis._get_all_permutations(chi - i, dimension-1)]
            return results
    
    def evaluate(self, X):
        return np.array([m.evaluate(X) for m in self.monomials])

class Regression:

    def __init__(self, X_mat, Y, chi=2, payoff_func=lambda x: np.max(np.sum(x)-100, 0)):
        assert len(X_mat.shape) == 2, "X in the regression should be a 2d matrix"
        self.dimension = len(X_mat[0])
        self.basis = Monomial_Basis(chi, self.dimension)
        
        self.index = np.array([i for i in range(len(X_mat)) if payoff_func(X_mat[i]) > 0])

        self.has_intrinsic_value = False if len(self.index) == 0 else True
        if not self.has_intrinsic_value: return

        target_X, target_Y = X_mat[self.index], Y[self.index]

        target_matrix_A = np.array([self.basis.evaluate(x) for x in target_X])
        self.coefficients = np.linalg.lstsq(target_matrix_A, target_Y, rcond=None)[0]

    def evaluate(self, X):
        """
        X: a numpy array of input data (e.g., asset prices)
        """
        if not self.has_intrinsic_value: raise RuntimeError("Least square failed due to ineiligible input")
        assert len(X) == self.dimension, "input vector X doesn't meet the regression dimension"
        monomial_terms = self.basis.evaluate(X)
        return np.sum(np.multiply(self.coefficients, monomial_terms))