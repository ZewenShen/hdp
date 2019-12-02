import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
import numpy as np
from timeit import default_timer as timer

class ExperimentResult:

    def __init__(self, r, N, results, times, errors, analytical_sol, cis=None):
        self.r = r
        self.N = N
        self.results = np.array(results)
        self.times = times
        self.analytical_sol = analytical_sol
        if errors:
            self.errors = np.array(errors)
            self.relative_errors = self.errors / analytical_sol
            self.orders = np.log2(self.errors[:-1] / self.errors[1:])
        else:
            self.errors = None
            self.orders = np.log2((errors[:-2] - errors[1:-1]) / (errors[1:-1] - errors[2:]))
        
        if cis:
            self.cis = np.array(cis)
            self.ci_orders = np.log2(self.cis[:-1] / self.cis[1:])
        else:
            self.cis = None

    def __str__(self):
        result = "N    approx    rel_err     ci     orders ci_orders  times\n"
        if self.cis is not None:
            result += "{:d}    {:.3f}    {:.1e}   {:.1e}      {:4s}    {:4s}  {:.2f}\n".format(self.r[0], self.results[0],\
                    self.relative_errors[0], self.cis[0], "NA", "NA", self.times[0])
            for i in range(1, len(self.r)):
                result += "{:d}    {:.3f}    {:.1e}   {:.1e}    {: .1f}    {: .1f}    {:.2f}\n".format(\
                    self.r[i], self.results[i], self.relative_errors[i], self.cis[i], self.orders[i-1], self.ci_orders[i-1], self.times[i])
        else:
            if self.errors is not None:
                result += "{:d}    {:.3f}    {:.1e}    NA    {:.2f}\n".format(self.r[0], self.results[0],\
                        self.relative_errors[0], self.times[0])
                for i in range(1, len(self.r)):
                    result += "{:d}    {:.3f}    {:.1e}    {: .1f}    {:.2f}\n".format(\
                        self.r[i], self.results[i], self.relative_errors[i], self.orders[i-1], self.times[i])
            else:
                result += "{:d}    {:.1e}    NA    {:.2f}\n".format(\
                        self.r[0], self.results[0], self.times[0])
                result += "{:d}    {:.1e}    NA    {:.2f}\n".format(\
                        self.r[1], self.results[1], self.times[1])
                for i in range(1, len(self.r)):
                    result += "{:d}    {:.1e}    {: .1f}    {:.2f}\n".format(\
                        self.r[i], self.results[i], self.orders[i-2], self.times[i])
        result += "Analytical Sol: " + "{:.6f}".format(self.analytical_sol) + '\n'
        return result

def FFTConvExperiment(analytical_sol, n_start, n_end, FFT_Euro, L_multiplier=30):
    results = []
    errors = []
    times = []
    r = range(n_start, n_end)
    N = 2**np.array(r)
    for i in r:
        start = timer()
        result = FFT_Euro.price(i * np.ones(FFT_Euro.dim, dtype=int), L_multiplier)
        end = timer()
        results.append(result)
        errors.append(abs(result - analytical_sol))
        times.append(end - start)
    return ExperimentResult(r, N, results, times, errors, analytical_sol)

def MCEuroExperiment(analytical_sol, n_start, n_end, MC_Euro, func_ver, ci=True):
    results = []
    cis = []
    errors = []
    times = []
    r = range(n_start, n_end)
    N = 2**np.array(r)
    for n in N:
        start = timer()
        if func_ver == "V2":
            result, ci_width = MC_Euro.priceV2(n, ci=ci)
        elif func_ver == "V4": # sobol
            result, ci_width = MC_Euro.priceV4(n, ci=ci)
        elif func_ver == "V5": # antithetic
            result, ci_width = MC_Euro.priceV5(n, ci=ci)
        elif func_ver == "V6":
            result, ci_width = MC_Euro.priceV6(n, ci=ci)
        else:
            raise RuntimeError("MCEuroExperiment: func_ver is not supported")
        end = timer()
        results.append(result)
        cis.append(ci_width)
        errors.append(abs(results[-1] - analytical_sol))
        times.append(end - start)
    return ExperimentResult(r, N, results, times, errors, analytical_sol, cis)
