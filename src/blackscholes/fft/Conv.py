import numpy as np
import itertools
from scipy.interpolate import interpn
# Convolution method for Black Scholes model

class ConvEuro:

    def __init__(self, payoff_func, S0_vec, T, ir, vol_vec, dividend_vec, corr_mat):
        self.payoff_func = payoff_func
        self.S0_vec = S0_vec
        self.T = T
        self.dim = len(vol_vec)
        self.ir = ir
        self.vol_vec = vol_vec
        self.dividend_vec = dividend_vec
        self.mu_vec = self.ir - self.dividend_vec - 0.5*self.vol_vec**2
        self.cov_mat = (self.vol_vec[np.newaxis].T @ self.vol_vec[np.newaxis]) * corr_mat
        self.price_mat = None

    def price(self, n_vec, L_multiplier=30, float32=False):
        N_vec = 2**n_vec
        self.N_vec = N_vec
        L_vec = self.vol_vec * self.T**0.5 * L_multiplier
        dy = L_vec / N_vec
        du = 2 * np.pi / L_vec
        grid = np.array([np.arange(N) for N in N_vec])
        if np.array_equal(N_vec, np.full(N_vec.shape, N_vec[0])):
            y = (grid - N_vec[np.newaxis].T / 2) * dy[np.newaxis].T
            u = (grid - N_vec[np.newaxis].T / 2) * du[np.newaxis].T
        else:
            y = (grid - N_vec / 2) * dy
            u = (grid - N_vec / 2) * du

        if float32: 
            self.S0_vec = self.S0_vec.astype(np.float32, copy=False)
            self.mu_vec = self.mu_vec.astype(np.float32, copy=False)
            self.cov_mat = self.cov_mat.astype(np.float32, copy=False)
            y = y.astype(np.float32, copy=False)
            u = u.astype(np.float32, copy=False)
        
        self.y = y
        k_vecs = ConvEuro.iterable_k_vec(N_vec)
        ys = y[np.arange(len(y)), k_vecs]
        self.points = self.S0_vec * np.exp(ys)
        V = self.payoff_func(self.S0_vec * np.exp(ys)).reshape(N_vec) # denormalize price
        del ys
        
        G = ConvEuro.G(k_vecs, N_vec)
        if float32: G = G.astype(np.float32, copy=False)
        us = u[np.arange(len(u)), k_vecs]
        del k_vecs

        phi = self.char_func(-us)
        del us

        fourier_price = phi * np.fft.ifftn(V * G)
        del V, G, phi

        self.price_mat = abs(np.fft.fftn(fourier_price).real * np.exp(-self.ir * self.T))
        return self.price_mat[tuple((N_vec/2).astype(int))]

    def get_all_price(self):
        return self.points, self.price_mat.flatten()

    def greeks(self):
        """
        return deltas and gammas
        """
        assert self.price_mat is not None, "Conv.delta: haven't priced yet"
        deltas = []
        gammas = []
        center_index = (self.N_vec/2).astype(int)
        for i in range(self.dim):
            center = self.S0_vec[i]
            left_index, right_index = np.copy(center_index), np.copy(center_index)
            left_index[i] -= 1; right_index[i] += 1
            left, right = np.exp(self.y[i][center_index[i]-1]), np.exp(self.y[i][center_index[i]+1])
            left *= self.S0_vec[i]; right *= self.S0_vec[i]
            h1, h2 = center - left, right - center
            left_price, right_price = self.price_mat[tuple(left_index)], self.price_mat[tuple(right_index)]
            center_price = self.price_mat[tuple(center_index)]
            deltas.append(-h2*left_price/(h1*(h1+h2)) + (h2-h1)*center_price/(h1*h2) + h1*right_price/((h1+h2)*h2))
            gammas.append(2*left_price/(h1*(h1+h2)) - 2*center_price/(h1*h2) + 2*right_price/((h1+h2)*h2))
        return np.array(deltas), np.array(gammas)

    def char_func(self, us):
        img = 1j * us @ self.mu_vec
        real = (us.dot(self.cov_mat)*us).sum(axis=1) / 2
        cf_val = np.exp(self.T*(img - real))
        return cf_val.reshape(self.N_vec)

    @staticmethod
    def G(k_vec, N_vec):
        index = np.logical_or(k_vec == N_vec - 1, k_vec == 0)
        return (0.5**np.sum(index, axis=1) * np.prod((-1)**k_vec, axis=1)).reshape(N_vec)
    
    @staticmethod
    def iterable_k_vec(N_vec):
        k_range = [range(n) for n in N_vec]
        return np.array(np.meshgrid(*k_range)).T.reshape(-1, len(N_vec))

class ConvEuro1d:

    def __init__(self, payoff_func, S0, T, ir, vol, dividend):
        self.payoff_func = payoff_func
        self.S0 = S0
        self.T = T
        self.ir = ir
        self.vol = vol
        self.dividend = dividend

    def price(self, n):
        N = 2**n
        L = self.vol * self.T**0.5 * 20
        dy, dx = L / N, L / N
        du = 2 * np.pi / L
        eps_y, eps_x = 0, 0
        grid = np.arange(N)
        grid_m = (-1)**grid
        y, x = eps_y + (grid - N / 2) * dy,  eps_x + (grid - N / 2) * dx
        u = (grid - N / 2) * du
        w = np.ones(N)
        w[0] = 0.5; w[-1] = 0.5

        V = self.payoff_func(self.S0 * np.exp(y))

        phi = self.log_char_func(-u)
        fourier_price = np.exp(1j * grid * (y[0] - x[0]) * du) * phi * np.fft.ifftn(V * w * grid_m)
        
        price = np.fft.fftn(fourier_price)
        C = price * np.exp(-self.ir * self.T + (1j * u * (y[0] - x[0]))) * grid_m
        return C[int(N/2)]

    def log_char_func(self, u):
        mu = self.ir - self.dividend
        logS0 = 0
        val = np.exp(1j * u * (logS0 + mu * self.T - np.log(self.char_func(-1j, 0))))
        return val * self.char_func(u, 0)

    def char_func(self, u, mu):
        return np.exp(1j * u * mu * self.T - self.vol**2 * u**2 * self.T / 2)