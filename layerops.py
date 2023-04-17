import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x) ** 2

def mse(y, ynext):
    return np.mean(np.power(y - ynext, 2))

def mse_prime(y, ynext):
    return 2 * (ynext - y) / y.size