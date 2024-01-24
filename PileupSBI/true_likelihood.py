import numpy as np

from scipy import signal

from simulators import RMF
from utils.pdfs import _poisson_pdf


def true_likelihood(rate:np.ndarray):
    """
    True likelihood for model in which pileup happens
    in bin space.
    """
    
    # rate[i] is \lambda(i)
    # which is the intensity of the source in each channel
    v = np.zeros_like(rate)
    total_rate = np.sum(rate)
    max_n = int(total_rate * 3 + 3)
    lam_tild = rate / total_rate
    m = len(lam_tild)
    lam_tild_conv = np.zeros((max_n, m))
    # 
    lam_tild_conv[0] = lam_tild
    for n in range(1, max_n):
        lam_tild_conv[n] = signal.fftconvolve(lam_tild_conv[n-1], lam_tild)[:m]
        lam_tild_conv[n, :n] = 0
    lam_tild_conv
    p_yt_i = np.sum(lam_tild_conv.T * _poisson_pdf(total_rate, np.array(range(max_n))), axis=1)
    return p_yt_i @ RMF
