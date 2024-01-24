import numpy as np

from scipy import special


def _poisson_pdf(lam, k):
    """Poisson pdf"""
    return lam ** k * np.exp(-k) / special.factorial(k)

def _normal_pdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * \
        np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
