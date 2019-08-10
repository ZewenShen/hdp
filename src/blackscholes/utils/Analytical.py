"""
@author: Abhirup Mishra
@Description: The class contains the methods for pricing options using Black-Scholes Formula
"""

from math import sqrt, pi
import scipy as sp
from scipy.special import erf
import numpy as np

class Analytical_Sol:
    """
    Analytical solution to 1D Black-Scholes Formula
    """
    #the constructor for the class
    def __init__(self, spot_price, strike_price, time_to_maturity, interest_rate, sigma, dividend_yield=0):
        
        ''' initializtion parameters '''
        self.spot_price = np.asarray(spot_price).astype(float)
        self.strike_price = strike_price
        self.time_to_maturity = time_to_maturity
        self.interest_rate = interest_rate
        self.sigma = sigma
        self.dividend_yield = dividend_yield                                         
            
    #private method for erf function    
    def bls_erf_value(self,input_number):
        erf_out = 0.5*(1 + erf(input_number/sqrt(2.0)))
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