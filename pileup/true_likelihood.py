import numpy as np
import pymc as pm

from scipy import signal

from txt_inputs.inputs import RMF
from utils.pdfs import _poisson_pdf, _normalise, _poisson_inverse_cdf
from utils.plot import hist_from_counts

def true_likelihood(rate:np.ndarray):
    """
    True likelihood for model in which pileup happens
    in bin space.
    """
    # rate[i] is \lambda(i)
    # which is the intensity of the source in each channel
    v = np.zeros_like(rate)
    total_rate = np.sum(rate)

    max_n = _poisson_inverse_cdf(total_rate, 0.001)
    p_Nt = _poisson_pdf(total_rate, np.array(range(max_n)))
    p_Nt = _normalise(p_Nt)

    lam_tild = rate / total_rate
    m = len(lam_tild)
    lam_tild_conv = np.zeros((max_n, m))
    lam_tild_conv[0] = lam_tild
    last_lam_tild_conv = lam_tild
    for n in range(1, max_n):
        last_lam_tild_conv = signal.fftconvolve(lam_tild_conv[n-1], lam_tild)
        lam_tild_conv[n] = last_lam_tild_conv[:m]
        #lam_tild_conv[n, :n] = 0
    lam_tild_conv.shape
    p_yt_i = np.sum(lam_tild_conv.T * p_Nt, axis=1)
    return p_yt_i @ RMF



if __name__ == '__main__':
    from spectralcomponents import Spectrum, GaussianEmissionLine, PowerLaw
    from simulators import simulator
    from matplotlib import pyplot as plt
    from time import time
    c1 = PowerLaw()
    c1args = (0.3, 1)
    c2 = GaussianEmissionLine()
    c2args = (0.1, 5, 0.05)
    spectrum = Spectrum(c1, c2)
    params = (c1args, c2args)
    start = time()
    data = simulator(spectrum, 10000, params, pileup='channels')  
    print (time() - start)
    hist_from_counts(data.astype(int), bins=30, density=True)
    rate = spectrum.get_rate(*params)
    true_like = true_likelihood(rate)
    plt.plot(rate @ RMF / np.sum(rate @ RMF), label='likelihood no pileup')
    plt.plot(true_like / np.sum(true_like), label='likelihood w pielup')
    plt.legend()