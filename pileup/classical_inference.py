import numpy as np

from scipy import signal

from pileup.inputs import RMF
from utils.pdfs import _poisson_pdf, _normalise, _poisson_inverse_cdf


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
    plt.show()
    lam_tild_conv.shape
    v = np.sum(lam_tild_conv.T * p_Nt, axis=1)[30:]
    return v


class DirectPosterior:
    def __init__(self, prior, spectrum, simulator):
        self.prior = prior
        self.spectrum = spectrum
        self.simulator = simulator

    def compute_true_likelihood(self, params, x0):
        """
        Compute the true likelihood of observing x0 given parameters.
        """
        rate = self.spectrum.get_rate(*params)
        likelihood = true_likelihood(rate)
        log_likelihood = np.log(likelihood)
        likelihood_of_x0 = 0
        for x in x0:
            likelihood_of_x0 += log_likelihood[int(x)]
        return likelihood_of_x0

    def sample_posterior(self, x0, num_samples, mu_init):
        """
        Sample from the approximated true posterior given observations x0.
        """
        mu_current = mu_init
        posterior_samples = np.zeros((num_samples, len(mu_init)))
        for i in range(num_samples):
            mu_proposal = self.prior.sample()
            likelihood_ratio = self.compute_true_likelihood(mu_proposal, x0) / self.compute_true_likelihood(mu_current, x0)
            acceptance_ratio = likelihood_ratio * (self.prior.pdf(mu_proposal) / self.prior.pdf(mu_current))
            if np.random.rand() < acceptance_ratio:
                mu_current = mu_proposal
            posterior_samples[i] = mu_current
        return posterior_samples



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