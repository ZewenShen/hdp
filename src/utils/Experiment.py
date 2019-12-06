import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/..")
import numpy as np
from timeit import default_timer as timer

class ExperimentResult:

    def __init__(self, r, N, results, times, errors, analytical_sol):
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
        

    def __str__(self):
        result = "N    approx    rel_err    orders  times\n"
        if self.errors is not None:
            result += "{:d}    {:.3f}    {:.1e}      NA    {:.2f}\n".format(self.r[0], self.results[0],\
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

def MCEuroExperiment(analytical_sol, n_start, n_end, MC_Euro, func_ver):
    results = []
    errors = []
    times = []
    r = range(n_start, n_end)
    N = 2**np.array(r)
    for n in N:
        start = timer()
        if func_ver == "V2":
            result = MC_Euro.priceV2(n)
        elif func_ver == "V4": # sobol
            result = MC_Euro.priceV4(n)
        elif func_ver == "V5": # antithetic
            result = MC_Euro.priceV5(n)
        elif func_ver == "V6":
            result = MC_Euro.priceV6(n)
        else:
            raise RuntimeError("MCEuroExperiment: func_ver is not supported")
        end = timer()
        results.append(result)
        errors.append(abs(results[-1] - analytical_sol))
        times.append(end - start)
    return ExperimentResult(r, N, results, times, errors, analytical_sol)

def MCEuroExperimentStd(n_start, n_end, sim_times, MC_Euro):
    print("N    std1    std2")
    r = range(n_start, n_end)
    N = 2**np.array(r)
    for path_num in N:
        std1, std2 = MCEuroExperimentStdHelper(sim_times, path_num, MC_Euro)
        print("{:d}    {:.3e}    {:.3e}".format(int(np.log2(path_num)), std1, std2))

def MCEuroExperimentStdHelper(N, path_num, MC_Euro):
    vanilla = []
    anti = []
    for n in range(N):
        np.random.seed(n)
        result1 = MC_Euro.priceV2(path_num)
        vanilla.append(result1)
        np.random.seed(n)
        result2 = MC_Euro.priceV5(path_num) # antithetic
        anti.append(result2)
    return np.std(vanilla, ddof=1), np.std(anti, ddof=1)