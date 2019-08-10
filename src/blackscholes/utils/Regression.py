import numpy as np

class Monomial:

    def __init__(self, a_vec):
        self.a_vec = a_vec
        self.dimension = len(a_vec)
    
    def evaluate(self, input_vec):
        assert self.dimension == len(input_vec), "Input data dimension is different from the monomial's dimension"
        return np.prod(np.power(input_vec, self.a_vec))

class Monomial_Basis:

    def __init__(self, chi, dimension):
        permutations = self._get_all_permutations(chi, dimension)
        self.monomial = list(map(lambda x: Monomial(x), permutations))

    def _get_all_permutations(self, chi, dimension):
        if chi == 0:
            return [[0]*dimension]
        elif dimension == 1:
            return [[i] for i in range(chi+1)]
        else:
            results = []
            for i in range(chi+1):
                result = map(lambda x: [i] + x, self._get_all_permutations(chi - i, dimension-1))
                results += list(result)
            return results


