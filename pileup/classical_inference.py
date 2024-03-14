from copy import copy
import numpy as np

from scipy.signal import fftconvolve

from inputs import RMF
from utils.pdfs import _poisson_pdf, _normalise, _poisson_inverse_cdf


def true_likelihood(rate: np.ndarray):
    """
    Computes the true likelihood for a model in which pileup happens
    in bin space. This function calculates the likelihood based on the
    given rate array, representing the intensity of the source in each channel.

    Parameters:
    - rate: np.ndarray, the rate (lambda) for each channel.

    Returns:
    - v: np.ndarray, the calculated true likelihood values.
    """
    # Calculate the total rate across all channels
    total_rate = np.sum(rate)
    
    # Determine the maximum number of events to consider based on the Poisson distribution
    max_n = _poisson_inverse_cdf(total_rate, 0.01)
    
    # Calculate the Poisson distribution PDF for required ns
    p_Nt = _poisson_pdf(total_rate, np.arange(max_n))
    p_Nt = _normalise(p_Nt)  # Normalize the Poisson PDF

    lam_tild = np.concatenate((np.zeros(30), rate / total_rate))
    m = len(lam_tild)
    
    # Initialize conv array
    lam_tild_conv = np.zeros((max_n, m * 2 - 1))
    lam_tild_conv[0, :m] = lam_tild
    
    # Perform convolution iteratively for each number of events
    for n in range(1, max_n):
        lam_tild_conv[n, :m * 2 - 1] = fftconvolve(lam_tild_conv[n - 1, :m], lam_tild, mode='full')[:m * 2 - 1]
    
    # Calculate the true likelihood by combining the convolution results with the Poisson probabilities
    v = np.sum(lam_tild_conv.T * p_Nt, axis=1)[30:]
    
    return v


def tensor_to_tuple(tensor):
    """
    Convert a tensor to a tuple of floats.

    Parameters:
    - tensor: torch.Tensor, the tensor to convert.

    Returns:
    - tuple, the converted tensor.
    """
    return tuple(copy(tensor).detach().cpu().numpy())


def tuple_to_tensor(tup):
    """
    Convert a tuple of floats to a tensor.

    Parameters:
    - tup: tuple, the tuple to convert.

    Returns:
    - torch.Tensor, the converted tuple.
    """
    return torch.tensor(tup)


class TruePosterior:
    def __init__(self, prior, spectrum, simulator):
        self.prior = prior
        self.spectrum = spectrum
        self.simulator = simulator

    def compute_true_likelihood(self, params, x0):
        """
        Compute the log likelihood of observing x0 given parameters.
        """
        rate = self.spectrum.get_rate(*params)
        log_likelihood = np.log(true_likelihood(rate))  # Assuming true_likelihood returns probability values
        log_likelihood_of_x0 = 0
        for x in x0:
            log_likelihood_of_x0 += log_likelihood[int(x)]
        return log_likelihood_of_x0


    def compute_posterior(self, params, x0):
        """
        Compute the posterior of the parameters given observations x0.
        """
        prior_prob = self.prior.log_prob(tuple_to_tensor(params))
        likelihood = self.compute_true_likelihood(params, x0)
        return prior_prob + likelihood
    
    def compute_grid_posterior(self, x0, grid):
        """
        Compute the posterior of the parameters given observations x0.
        """
        posterior = np.zeros(grid.shape)
        for i, mu in enumerate(grid):
            posterior[i] = self.compute_posterior(mu, x0)
        return posterior

    def sample_posterior(self, x0, num_samples, mu_init):
        """
        Sample from the approximated true posterior given observations x0 using log likelihoods.
        Implements a Metropolis-Hastings proposal to sample with MCMC.
        """
        mu_current = mu_init
        posterior_samples = np.zeros((num_samples, int(np.sum(len(mu) for mu in mu_init))))
        log_likelihood_current = self.compute_true_likelihood(mu_current, x0)
        for i in range(num_samples):
            mu_proposal = self.prior.sample() # torch.Tensor
            log_likelihood_proposal = self.compute_true_likelihood((tensor_to_tuple(mu_proposal),), x0) # float = log_likelihood(tuple(tuple), np.array)
            log_likelihood_ratio = log_likelihood_proposal - log_likelihood_current
            log_prior_ratio = self.prior.log_prob(mu_proposal) - self.prior.log_prob(tuple_to_tensor(mu_current))
            log_acceptance_ratio = log_likelihood_ratio + log_prior_ratio
            if np.log(np.random.rand()) < log_acceptance_ratio:
                mu_current = mu_proposal
                log_likelihood_current = log_likelihood_proposal  # Update the current log likelihood
            posterior_samples[i] = np.array(mu_current)
        return posterior_samples


if __name__ == '__main__':
    from spectralcomponents import Spectrum, GaussianEmissionLine, PowerLaw
    from simulators import Simulator
    from matplotlib import pyplot as plt
    from time import time
    c1 = PowerLaw()
    c1args = (0.1, 1)
    #c2 = GaussianEmissionLine()
    #c2args = (0.1, 10, 0.05)
    spectrum = Spectrum(c1)
    params = (c1args,)
    start = time()
    simulator = Simulator(spectrum, 100000, pileup='bins')  
    data = simulator(params)
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

    import torch
    from sbi_tools import BoxUniform
    prior = BoxUniform(low=torch.tensor([0.01, 0.01]), high=torch.tensor([1, 1]))
    posterior = TruePosterior(prior, spectrum, simulator)
    posterior_samples = posterior.sample_posterior(data, 1000, ((0.5, 0.5),))