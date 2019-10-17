import numpy as np
import itertools
from scipy import interpolate

# Convolution method for Black Scholes model

class ConvEuro:

    def __init__(self, payoff_func, T, ir, vol_vec, dividend_vec, corr_mat):
        assert hasattr(payoff_func, 'strike'), 'Meta info of the payoff function lost: strike'
        self.payoff_func = payoff_func
        self.strike = payoff_func.strike
        self.T = T
        self.dim = len(vol_vec)
        self.ir = ir
        self.vol_vec = vol_vec
        self.dividend_vec = dividend_vec
        self.corr_mat = corr_mat

    def pricing_func(self, N_vec, freq_grid_size_vec):
        time_grid_size_vec = 2 * np.pi / (N_vec * freq_grid_size_vec)
        print(freq_grid_size_vec, time_grid_size_vec)
        freq_grid_start = -0.5 * N_vec * freq_grid_size_vec
        time_grid_start = -0.5 * N_vec * time_grid_size_vec
        print(freq_grid_start, time_grid_start)
        b = time_grid_start+np.arange(N_vec[0])*time_grid_size_vec
        a = np.exp(b)
        
        V, G, phi = np.zeros(N_vec), np.zeros(N_vec), np.zeros(N_vec, dtype=np.complex)
        for k_vec in ConvEuro.iterable_k_vec(N_vec):
            k_vec = np.array(k_vec)
            omega = freq_grid_start + k_vec * freq_grid_size_vec
            y = time_grid_start + k_vec * time_grid_size_vec
            V[k_vec] = self.payoff_func(np.exp(y)) # denormalize price
            G[k_vec] = ConvEuro.G(k_vec, N_vec)
            phi[k_vec] = self.char_func(-omega)
        print(V, G)
        fourier_price = phi * np.fft.ifftn(V * G)
        
        # print(phi[5120], V[5120], G[5120])
        # print(fourier_price, fourier_price*np.array([1, -1]*5120))
        price = np.fft.fftn(fourier_price*np.array([1, -1]*256))
        # print(fourier_price, price[255:260])
        # print(b[5120], a[5120])
        return price * np.exp(-self.ir * self.T)

    def char_func(self, omega_vec):
        mu_vec = self.T * (self.ir - self.dividend_vec - 0.5*self.vol_vec**2)
        cf_val1 = 1j * np.dot(mu_vec, omega_vec)
        cf_val2 = 0
        for j in range(self.dim):
            for k in range(self.dim):
                cf_val2 += self.corr_mat[j, k] * self.vol_vec[j] * self.vol_vec[k] * omega_vec[j] * omega_vec[k]
        cf_val = np.exp(cf_val1 - self.T*cf_val2)
        return cf_val

    @staticmethod
    def R(k_vec, N_vec):
        index = np.logical_or(k_vec == N_vec - 1, k_vec == 0)
        result = np.ones(len(k_vec))
        result[index] = 0.5
        return result

    @staticmethod
    def Z(k_vec, N_vec):
        return np.prod(ConvEuro.R(k_vec, N_vec))

    @staticmethod
    def G(k_vec, N_vec):
        return ConvEuro.Z(k_vec, N_vec) * np.prod((-1)**k_vec)
    
    @staticmethod
    def iterable_k_vec(N_vec):
        k_range = [range(n) for n in N_vec]
        return list(itertools.product(*k_range))
