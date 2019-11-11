from math import sqrt, pi
import scipy as sp
from scipy.special import erf
from scipy.stats import norm
import numpy as np

class GeometricAvg:
    def __init__(self, dim, spot, strike, T, ir, vol_vec, dividend, corr_mat):
        self.spot = spot
        self.strike = strike
        self.T = T
        self.ir = ir
        if isinstance(vol_vec, float):
            self.vol_vec = vol_vec * np.ones(dim)
        else:
            self.vol_vec = vol_vec
        self.dim = dim
        self.dividend = dividend
        if isinstance(corr_mat, float):
            self.corr_mat = np.full((dim, dim), corr_mat)
            np.fill_diagonal(self.corr_mat, 1)
        else:
            self.corr_mat = corr_mat

    def european_option_price(self):
        sigma = (self.vol_vec @ self.corr_mat @ self.vol_vec)**0.5 / self.dim
        F = np.prod(self.spot)**(1/self.dim) *\
             np.exp((self.ir - self.dividend - np.sum(self.vol_vec**2)/(2*self.dim) + sigma**2/2)*self.T)
        d1 = (np.log(F/self.strike) + sigma**2*self.T/2)/(sigma*self.T**0.5)
        d2 = d1 - sigma * self.T**0.5
        return np.exp(-self.ir * self.T) * (F * norm.cdf(d1) - self.strike * norm.cdf(d2))


"""
@author: Abhirup Mishra
@Description: The class contains the methods for pricing options using Black-Scholes Formula
"""
class Analytical_Sol:
    """
    Analytical solution to 1D Black-Scholes Formula
    """
    #the constructor for the class
    def __init__(self, spot_price, strike_price, time_to_maturity, interest_rate, sigma, dividend_yield=0):
        
        ''' initializtion parameters '''
        self.spot_price = np.asarray(spot_price).astype(float)
        
        #checking if strike price is an array or not
        if(not(hasattr(strike_price, "__len__"))):
            self.strike_price = np.ones(self.spot_price.size)*strike_price
        else:
            self.strike_price = strike_price

        #checking if time to maturity is an array or not
        if(not(hasattr(time_to_maturity, "__len__"))):
            self.time_to_maturity = np.ones(self.spot_price.size)*time_to_maturity
        else:
            self.time_to_maturity = time_to_maturity

        #checking if interest rate is an array or not
        if(not(hasattr(interest_rate, "__len__"))):
            self.interest_rate = np.ones(self.spot_price.size)*interest_rate
        else:
            self.interest_rate = interest_rate

        #checking if volatility is an array or not
        if(not(hasattr(sigma, "__len__"))):
            self.sigma = np.ones(self.spot_price.size)*sigma
        else:
            self.sigma = sigma
        
        #checking if dividend yield is an array or not
        if(not(hasattr(dividend_yield, "__len__"))):
            self.dividend_yield = np.ones(self.spot_price.size)*dividend_yield
        else:
            self.dividend_yield = dividend_yield                                         
            
    #private method for erf function    
    def bls_erf_value(self, input_number):
        erf_out = 0.5*(1 + erf(input_number / sqrt(2.0)))
        return erf_out
    
    #vectorized method to price call option
    def european_option_price(self):
        
        "Price of the call option"
        "the vectorized method can compute price of multiple options in array"
        numerator = sp.add(sp.log(sp.divide(self.spot_price,self.strike_price)), sp.multiply((self.interest_rate - self.dividend_yield + 0.5*sp.power(self.sigma,2)),self.time_to_maturity))
        d1 = sp.divide(numerator,sp.prod([self.sigma,sp.sqrt(self.time_to_maturity)],axis=0))
        d2 = sp.add(d1, -sp.multiply(self.sigma,sp.sqrt(self.time_to_maturity)))
        
        ecall = sp.product([self.spot_price, self.bls_erf_value(d1), sp.exp(sp.multiply(-self.dividend_yield,self.time_to_maturity))],axis=0) \
                          - sp.product([self.strike_price,self.bls_erf_value(d2),sp.exp(-sp.multiply(self.interest_rate,self.time_to_maturity))],axis=0)
        
        eput = sp.product([-self.spot_price, self.bls_erf_value(-d1), sp.exp(sp.multiply(-self.dividend_yield,self.time_to_maturity))],axis=0) \
                          + sp.product([self.strike_price,self.bls_erf_value(-d2),sp.exp(-sp.multiply(self.interest_rate,self.time_to_maturity))],axis=0)
        return ecall, eput