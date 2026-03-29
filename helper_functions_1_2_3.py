# general packages across question 1, 2, and 3
# For question everything is done in google colab
import numpy as np
import numba as nb # this one is used to speed up MC in question 1, could also be used in question 2 and 3, but as they are relatively small, no need
from numpy       import pi, sqrt, log, exp
from scipy.stats import norm
import scipy
import matplotlib.pyplot as plt
the_seed = 123
rng = np.random.default_rng(seed=the_seed)

# Helper functions across question 1,2, and3 


# Quesiton 1, Helper functions
@nb.njit
def random_in_square():
    """Returns a random position in the square [-1,1)x[-1,1)."""
    return rng.uniform(-1,1,2) 

@nb.njit
def is_in_circle(x):
    return np.dot(x, x) < 1.0

@nb.njit
def is_in_square(x):
    return (x[0] > -1.0 and x[0] < 1.0 and x[1] > -1.0 and x[1] < 1.0)

@nb.njit
def simulate_number_of_hits(N):
    """Simulates number of hits in case of N trials in the pebble game."""
    number_hits = 0
    for i in range(N):
        position = random_in_square()
        if is_in_circle(position):
            number_hits += 1
    return number_hits

@nb.njit
def circle_throw(delta = 0.4):
    '''Bassed on: https://stackoverflow.com/questions/62046851/generate-uniformly-distributed-points-in-a-circle'''
    if delta == 0.0:
        return np.array([0.0,0.0])
    else:
        theta = np.random.uniform(0.0, 2.0 * np.pi)
        radius = np.sqrt(np.random.uniform(0.0, delta * delta))
        return np.array([radius * np.cos(theta),radius * np.sin(theta)])


# Quesiton 2, Helper functions


# Quesiton 3, Helper functions

def norm_cdf(x):
    if not isinstance(x, np.ndarray):
        xr = x.real
        xi = x.imag
        if abs(xi) > 1.0e-10:
            raise ValueError('imag(x) too large in norm_cdf(x)')

        ncf = norm.cdf(xr)
        if abs(xi) > 0:
            ncf = ncf + 1.0j*xi*norm.pdf(xr)
    else:
        xr = np.real(x)
        xi = np.imag(x)
        if any(abs(xi) > 1.0e-10):
            raise ValueError('imag(x) too large in norm_cdf(x)')

        ncf = norm.cdf(xr)
        if any(abs(xi) > 0):
            ncf = ncf + 1.0j*xi*norm.pdf(xr)

    return ncf

def f_ex_3(x):
    return x*np.cos(np.pi * x)

def MC_est(U):
    return np.mean(f_ex_3(U),axis=1)
    
def exercise_3_2(m=1000, n=1000):
    U = rng.uniform(size=(m, n))
    return MC_est(U)

def european_call(r=0.05,sigma=0.2,T=1,S=100,K=100,opt='value'):

    S  = S + 1.0e-100     # avoids problems with S=0
    K  = K + 1.0e-100     # avoids problems with K=0

    d1 = ( log(S) - log(K) + (r+0.5*sigma**2)*T ) / (sigma*sqrt(T))
    d2 = ( log(S) - log(K) + (r-0.5*sigma**2)*T ) / (sigma*sqrt(T))

    if opt == 'value':
        V = S*norm_cdf(d1) - exp(-r*T)*K*norm_cdf(d2)
    elif opt == 'delta':
        V = norm_cdf(d1)
    elif opt == 'gamma':
        V = exp(-0.5*d1**2) / (sigma*sqrt(2*pi*T)*S)
    elif opt == 'vega':
        V =             S*(exp(-0.5*d1**2)/sqrt(2*pi))*( sqrt(T)-d1/sigma) \
            - exp(-r*T)*K*(exp(-0.5*d2**2)/sqrt(2*pi))*(-sqrt(T)-d2/sigma)

    else:
        raise ValueError('invalid value for opt -- must be "value", "delta", "gamma", "vega"')

    return V