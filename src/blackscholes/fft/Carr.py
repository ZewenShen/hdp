import numpy as np
from scipy import interpolate

# Carr/Madan method for Black Scholes model


class CarrEuroCall1d:

    def __init__(self, T, S0, ir, vol, alpha=1.5):
        self.T = T
        self.x0 = np.log(S0) # stock price at time 0
        self.ir = ir
        self.vol = vol
        self.alpha = alpha

    def char_func(self, v):
        cf_val = np.exp(1j * (self.x0 + (self.ir - self.vol ** 2 / 2) * self.T) * v \
                        - self.vol ** 2 * v ** 2 * self.T / 2)
        return cf_val

    def damped_fourier_price(self, v):
        alpha = self.alpha
        C = np.exp(-self.ir * self.T)
        Psi = C * self.char_func(v - 1j*(alpha+1)) / ((alpha+1j*v) * (alpha+1j*v+1))
        return Psi

    def pricing_func(self, N, strike_grid_size):
        int_grid_size = 2*np.pi/(N*strike_grid_size)
        beta = self.x0 - strike_grid_size*N/2 # lower bound
        k_vec = np.array([beta + i * strike_grid_size for i in range(N)])
        v_vec = np.array([i * int_grid_size for i in range(N)])
        Psi_vec = self.damped_fourier_price(v_vec)
        FFTfunc = Psi_vec * np.exp(-1j * beta * v_vec) * simpson(N, int_grid_size)
        FFTinv = np.fft.fft(FFTfunc).real
        CT = np.exp(-self.alpha*k_vec) * FFTinv / np.pi
        self.k_vec = np.exp(k_vec)
        self.price = CT
        return interpolate.interp1d(self.k_vec, CT, 'cubic')

def simpson(N, int_grid_size):
    delta = np.zeros(N, dtype=np.float)
    delta[0] = 1
    j = np.arange(1, N + 1, 1)
    simpson_coeff = int_grid_size*(3 + (-1) ** j - delta) / 3
    return simpson_coeff

