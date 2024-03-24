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
    rate = rate @ RMF
    total_rate = np.sum(rate)

    # Determine the maximum number of events to consider based on the Poisson distribution
    max_n = _poisson_inverse_cdf(total_rate, 0.01)

    # Calculate the Poisson distribution PDF for required ns
    p_Nt = _poisson_pdf(total_rate, np.arange(max_n))
    p_Nt = _normalise(p_Nt)  # Normalise the Poisson PDF

    lam_tild = np.concatenate((np.zeros(30), rate / total_rate))
    m = len(lam_tild)

    # Initialize conv array
    lam_tild_conv = np.zeros((max_n, m * 2 - 1))
    lam_tild_conv[0, :m] = lam_tild

    # Convolution iteratively for each number of events
    for n in range(1, max_n):
        lam_tild_conv[n, :m * 2 - 1] = fftconvolve(lam_tild_conv[n - 1, :m], lam_tild, mode='full')[:m * 2 - 1]

    # Calculate the true likelihood by combining the convolution results with the Poisson probabilities
    v = np.sum(lam_tild_conv.T * p_Nt, axis=1)[:1024]
    v[0] += 1 - np.sum(v[1:])
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
    def __init__(self, prior, spectrum, obs):
        self.prior = prior
        self.spectrum = spectrum
        self.obs = obs

    def compute_true_likelihood(self, params):
        """
        Compute the log likelihood of observing x0 given parameters.
        """
        x0 = self.obs
        rate = self.spectrum.get_rate(params)
        log_likelihood = np.log(true_likelihood(rate))  # Assuming true_likelihood returns probability values
        log_likelihood += np.min(log_likelihood)  # Shift the log likelihood to avoid numerical issues
        print (log_likelihood)
        log_likelihood_of_x0 = 0
        for x in x0:
            log_likelihood_of_x0 += log_likelihood[int(x)]
        return log_likelihood_of_x0


    def compute_log_posterior(self, params):
        """
        Compute the posterior of the parameters given observations x0.
        """
        log_prior = self.prior.log_prob(tuple_to_tensor(params))
        log_likelihood = np.log(self.compute_true_likelihood(params))
        return log_prior + log_likelihood


    def compute_grid_posterior(self, params1, params2, *args):
        """
        Compute the posterior of the parameters given observations x0.
        """
        x0 = self.obs
        if args is not None:
            print ('Currently not supporting more than 2 parameters')
            return
        
        posterior = np.zeros((len(params1), len(params2)))
        for i, param1 in enumerate(params1):
            for j, param2 in enumerate(params2):
                posterior[i, j] = self.compute_log_posterior((param1, param2), x0)
        # now normalise so that np.exp(posterior) will sum to 1
        posterior = np.exp(posterior - np.max(posterior))
        return posterior


    def sample_posterior(self, num_samples, mu_init):
        """
        Sample from the approximated true posterior given observations x0 using log likelihoods.
        Implements a Metropolis-Hastings proposal to sample with MCMC.
        """
        x0 = self.obs
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
    from spectralcomponents import PowerLaw, Spectrum
    from simulators import Simulator
    from sbi.utils.torchutils import BoxUniform
    from torch import tensor

    c1 = PowerLaw()
    spectrum = Spectrum(c1)
    params = (0.5, 0.3)
    prior = BoxUniform(low=tensor([0.1, 0.1]), high=tensor([1, 2]))
    simulator =  Simulator(spectrum, 100, pileup='channels')
    data = simulator(tensor(params))
    posterior = TruePosterior(prior, spectrum, data)
    print (posterior.compute_true_likelihood((0.5, 0.2)))
