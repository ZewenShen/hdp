import numpy as np
import itertools
from scipy import interpolate

# Convolution method for Black Scholes model

class ConvEuro:

    def __init__(self, payoff_func, S0_vec, T, ir, vol_vec, dividend_vec, corr_mat):
        assert hasattr(payoff_func, 'strike'), 'Meta info of the payoff function lost: strike'
        self.payoff_func = payoff_func
        self.S0_vec = S0_vec
        self.strike = payoff_func.strike
        self.T = T
        self.dim = len(vol_vec)
        self.ir = ir
        self.vol_vec = vol_vec
        self.dividend_vec = dividend_vec
        self.corr_mat = corr_mat

    def pricing_func(self, N_vec):
        L_vec = self.vol_vec * self.T**0.5
        dy = L_vec / N_vec
        dx = dy
        du = 2 * np.pi / L_vec
        eps_y = np.zeros_like(N_vec)
        eps_x = np.zeros_like(N_vec)
        grid = np.array([np.arange(N) for N in N_vec])
        y = eps_y + (grid - N_vec / 2) * dy
        x = eps_x + (grid - N_vec / 2) * dx
        u = (grid - N_vec / 2) * du

        V, G, phi = np.zeros(N_vec), np.zeros(N_vec), np.zeros(N_vec, dtype=np.complex)
        for k_vec in ConvEuro.iterable_k_vec(N_vec):
            k_vec = np.array(k_vec)
            V[k_vec] = self.payoff_func(self.S0_vec * np.exp(y[k_vec])) # denormalize price
            G[k_vec] = ConvEuro.G(k_vec, N_vec)
            phi[k_vec] = self.char_func(-u[k_vec])
        # print(V* G, V, G)
        fourier_price = np.exp(1j * grid * (y[:, 0] - x[:, 0]) * du) * phi * np.fft.ifftn(V * G)
        
        price = np.fft.fftn(fourier_price)
        # print(fourier_price, price[255:260])
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


class ConvEuro1d:

    def __init__(self, payoff_func, S0, T, ir, vol, dividend):
        assert hasattr(payoff_func, 'strike'), 'Meta info of the payoff function lost: strike'
        self.payoff_func = payoff_func
        self.strike = payoff_func.strike
        self.S0 = S0
        self.T = T
        self.ir = ir
        self.vol = vol
        self.dividend = dividend

    def pricing_func(self, N_vec):
        L_vec = self.vol_vec * self.T**0.5
        dy = L_vec / N_vec
        dx = dy
        du = 2 * np.pi / L_vec
        eps_y = np.zeros_like(N_vec)
        eps_x = np.zeros_like(N_vec)
        grid = np.array([np.arange(N) for N in N_vec])
        y = eps_y + (grid - N_vec / 2) * dy
        x = eps_x + (grid - N_vec / 2) * dx
        u = (grid - N_vec / 2) * du

        V, G, phi = np.zeros(N_vec), np.zeros(N_vec), np.zeros(N_vec, dtype=np.complex)
        for k_vec in ConvEuro.iterable_k_vec(N_vec):
            k_vec = np.array(k_vec)
            V[k_vec] = self.payoff_func(self.S0_vec * np.exp(y[k_vec])) # denormalize price
            G[k_vec] = ConvEuro.G(k_vec, N_vec)
            phi[k_vec] = self.char_func(-u[k_vec])
        # print(V* G, V, G)
        fourier_price = np.exp(1j * grid * (y[:, 0] - x[:, 0]) * du) * phi * np.fft.ifftn(V * G)
        
        price = np.fft.fftn(fourier_price)
        # print(fourier_price, price[255:260])
        return price * np.exp(-self.ir * self.T)

    def char_func(self, omega):
        mu_vec = self.T * (self.ir - self.dividend - 0.5*self.vol**2)
        cf_val1 = 1j * np.dot(mu_vec, omega)
        
        return cf_val