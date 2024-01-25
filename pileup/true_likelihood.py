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
    
    total_rate = np.sum(rate)
    max_n = _poisson_inverse_cdf(total_rate, 0.00001)
    p_Nt = _poisson_pdf(total_rate, np.array(range(max_n)))
    p_Nt = _normalise(p_Nt)

    lam_tild = np.concatenate((np.zeros(30), rate / total_rate))
    m = len(lam_tild)
    lam_tild_conv = np.zeros((max_n, m * 2 - 1))
    lam_tild_conv[0, :m] = lam_tild
    last_lam_tild_conv = lam_tild
    for n in range(1, max_n):
        last_lam_tild_conv = signal.fftconvolve(last_lam_tild_conv, lam_tild)
        lam_tild_conv[n] = last_lam_tild_conv[:m * 2 - 1]
        plt.plot(last_lam_tild_conv)
        print (last_lam_tild_conv.shape)
        #lam_tild_conv[n, :n] = 0
    plt.show()
    lam_tild_conv.shape
    v = np.sum(lam_tild_conv.T * p_Nt, axis=1)[30:]
    v
    return v



if __name__ == '__main__':
    from spectralcomponents import Spectrum, GaussianEmissionLine, PowerLaw
    from simulators import simulator
    from matplotlib import pyplot as plt
    from time import time
    c1 = PowerLaw()
    c1args = (0.2, 1)
    #c2 = GaussianEmissionLine()
    #c2args = (0.1, 10, 0.05)
    spectrum = Spectrum(c1)
    params = (c1args,)
    start = time()
    data = simulator(spectrum, 100000, params, pileup='channels')  
    print (time() - start)
    
    rate = spectrum.get_rate(*params)
    start = time()
    true_like = true_likelihood(rate)[:1070]
    print (time() - start)
    print (true_like.sum())
    plt.plot((rate / np.sum(rate) @ RMF), label='likelihood no pileup')
    plt.plot((true_like / np.sum(true_like) @ RMF), label='likelihood w pielup')
    plt.hist(data[data>0], bins=50, density=True)
    plt.legend()