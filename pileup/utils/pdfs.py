import numpy as np

from scipy import special
from scipy.stats import poisson


def _poisson_pdf(lam, k):
    """Poisson pdf"""
    return lam ** k * np.exp(-k) / special.factorial(k)

def _normal_pdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * \
        np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def _normalise(truncated_pdf):
    return truncated_pdf / np.sum(truncated_pdf)

def _poisson_inverse_cdf(rate, epsilon):
    """
    Returns n such that P(X>n)<epsilon.
    Assumes X~Poisson(rate)
    """
    n = 0  # Start checking from 0
    # Calculate the probability P(X > n) and check if it is less than epsilon
    while True:
        # 1 - CDF is the probability that X is greater than n
        if 1 - poisson.cdf(n, rate) < epsilon:
            return n
        n += 1
