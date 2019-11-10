import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
import numpy as np
from timeit import default_timer as timer

class ExperimentResult:

    def __init__(self, N, results, times, errors, analytical_sol):
        self.N = N
        self.results = np.array(results)
        self.times = times
        if errors:
            self.errors = np.array(errors)
            self.relative_errors = self.errors / analytical_sol
            self.orders = np.log2(self.errors[:-1] / self.errors[1:])
        else:
            self.errors = None
            self.orders = np.log2((errors[:-2] - errors[1:-1]) / (errors[1:-1] - errors[2:]))

    def __str__(self):
        result = "N    rel_err    orders    times\n"
        if self.errors is not None:
            result += "{:d}    {:.1e}    NA    {:.1f}\n".format(\
                    self.N[0], self.relative_errors[0], self.times[0])
            for i in range(1, len(self.N)):
                result += "{:d}    {:.1e}    {:.1f}    {:.1f}\n".format(\
                    self.N[i], self.relative_errors[i], self.orders[i-1], self.times[i])
        else:
            result += "{:d}    {:.1e}    NA    {:.1f}\n".format(\
                    self.N[0], self.relative_errors[0], self.times[0])
            result += "{:d}    {:.1e}    NA    {:.1f}\n".format(\
                    self.N[1], self.relative_errors[1], self.times[1])
            for i in range(1, len(self.N)):
                result += "{:d}    {:.1e}    {:.1f}    {:.1f}\n".format(\
                    self.N[i], self.relative_errors[i], self.orders[i-2], self.times[i])
        return result

def FFTConvExperiment(analytical_sol, n_start, n_end, FFT_Euro, L_multiplier=30):
    results = []
    errors = []
    times = []
    r = range(n_start, n_end)
    N = 2**np.array(r)
    for i in r:
        start = timer()
        result = FFT_Euro.price(i * np.ones(FFT_Euro.solver.dim, dtype=int), L_multiplier)
        end = timer()
        results.append(result)
        errors.append(abs(result - analytical_sol))
        times.append(end - start)
    return ExperimentResult(N, results, times, errors, analytical_sol)

def MCEuroExperiment(analytical_sol, n_start, n_end, MC_Euro):
    results = []
    errors = []
    times = []
    r = range(n_start, n_end)
    N = 2**np.array(r)
    for n in N:
        start = timer()
        result = MC_Euro.priceV2(n)
        end = timer()
        results.append(result)
        errors.append(abs(results[-1] - analytical_sol))
        times.append(end - start)
    return ExperimentResult(N, results, times, errors, analytical_sol)

