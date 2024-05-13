from copy import copy
import numpy as np
import torch
from tqdm import tqdm

from scipy.signal import fftconvolve

from inputs import RMF
from utils.pdfs import _poisson_pdf, _normalise, _poisson_inverse_cdf


def true_likelihood_bins(rate: np.ndarray, alpha: float = 0.5):
    """
    Computes the true likelihood for a model in which pileup happens
    in bin space. This function calculates the likelihood based on the
    given rate array, representing the intensity of the source in each channel.

    Parameters:
    - rate: np.ndarray, the rate (lambda) for each channel.

    Returns:
    - v: np.ndarray, the calculated true likelihood values.
    """
    total_rate = np.sum(rate)

    # determine the maximum number of events to consider based on the Poisson distribution
    #max_n = _poisson_inverse_cdf(total_rate, 0.01)
    max_n = 5
    # calculate the Poisson distribution PDF for required ns
    p_Nt = _poisson_pdf(total_rate, np.arange(max_n))
    p_Nt = _normalise(p_Nt)  # normalise the Poisson PDF

    lam_tild = np.concatenate((np.zeros(30), rate / total_rate))
    m = len(lam_tild)

    lam_tild_conv = np.zeros((max_n, m))
    lam_tild_conv[0, :] = lam_tild

    # convolution iteratively for each number of events
    for n in range(1, max_n):
        lam_tild_conv[n, :] = fftconvolve(lam_tild_conv[n - 1, :m], lam_tild, mode='same')
        lam_tild_conv[n, :n] = 0
    v = np.sum(lam_tild_conv.T * p_Nt, axis=1)[30:]
    v = v.astype(np.float64) @ RMF
    v0 = 1 - np.sum(v) # mthmaticaly equivalent to the above??
    #print (f"pr no ev {prob_of_no_event}")
    v = np.concatenate(([v0], v))
    return v.astype(np.float64)


def true_likelihood_channels(rate: np.ndarray, alpha: float = 0.5):
    rate = rate @ RMF
    total_rate = np.sum(rate)

    # determine the maximum number of events to consider based on the Poisson distribution
    max_n = _poisson_inverse_cdf(total_rate, 0.01)
    # print (max_n)

    # calculate the Poisson distribution PDF for required ns
    p_Nt = _poisson_pdf(total_rate, np.arange(max_n + 2))
    p_Nt = _normalise(p_Nt)  # normalise the Poisson PDF

    lam_tild = rate / total_rate
    m = len(lam_tild)

    size_needed = m + (m - 1) * max_n  # Correct maximum size needed
    lam_tild_conv = np.zeros((max_n + 1, size_needed))

    lam_tild_conv[0, :m] = lam_tild

    for n in range(1, max_n + 1):
        # full convolution expands by (m-1) each step
        current_result_size = m + (m - 1) * n
        lam_tild_conv[n, :current_result_size] = fftconvolve(lam_tild_conv[n - 1, :m + (m - 1) * (n - 1)], lam_tild, mode='full')
        lam_tild_conv[n, :n] = 0

    # extract the valid portion of the convolution
    valid_convolution = lam_tild_conv[:, :m]
    alphas = alpha ** np.arange(max_n + 1)
    v = np.sum(valid_convolution.T * (p_Nt[1:] * alphas), axis=1)
    v0 = 1 - np.sum(v)
    v = np.concatenate(([v0], v))

    return v.astype(np.float64)


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
    def __init__(self, prior, spectrum, obs, pileup='channels'):
        self.prior = prior
        self.spectrum = spectrum
        self.obs = obs
        self.pileup = pileup

    def compute_true_likelihood(self, params):
        """
        Compute the log likelihood of observing x0 given parameters.
        """
        x0 = self.obs
        rate = self.spectrum.get_rate(params)
        if self.pileup == 'bins':
            likelihood = true_likelihood_bins(rate)
        elif self.pileup == 'channels':
            likelihood = true_likelihood_channels(rate)
        log_likelihood_of_x0 = 0
        for x in x0:
            log_likelihood_of_x0 += np.log(likelihood[x])
        if np.isnan(log_likelihood_of_x0):
            return -np.inf
        return log_likelihood_of_x0


    def compute_log_posterior(self, params):
        """
        Compute the posterior of the parameters given observations x0.
        """
        log_prior = self.prior.log_prob(tuple_to_tensor(params))
        log_likelihood = self.compute_true_likelihood(params)
        return log_likelihood + log_prior


    def compute_grid_posterior(self, params1, params2):
        """
        Compute the posterior of the parameters given observations x0.
        """
        posterior = np.zeros((len(params1), len(params2)))
        for i, param1 in enumerate(tqdm(params1, desc='Computing posterior', leave=False)):
            for j, param2 in enumerate(params2):
                posterior[i, j] = self.compute_log_posterior((param1, param2))
        # now normalise so that np.exp(posterior) will sum to 1
        posterior = np.exp(posterior - np.max(posterior))
        return posterior / np.sum(posterior)


    def find_maximum_likelihood(self, initial_params, bounds):
        """
        Find the maximum likelihood parameters given observations x0.
        """
        from scipy.optimize import minimize
        result = minimize(
            lambda theta: -self.compute_true_likelihood(theta), 
            initial_params, 
            method='L-BFGS-B')
        self.maximum_likelihood = result.x
        return result.x
    
    def sample_with_MCMC(self, true_params):
        """
        Sample from the posterior using MCMC.
        We wish to construct a markov chain which uses the .compute_true_likelihood above.
        """
        initial_guess  = self.find_maximum_likelihood(true_params, None)
        from scipy.stats import multivariate_normal
        from scipy.optimize import minimize
        from scipy.stats import uniform
        from scipy.stats import norm
        from scipy.stats import poisson
        from scipy.stats import gamma

        def log_prior(params):
            return self.prior.log_prob(tuple_to_tensor(params))
        
        def log_likelihood(params):
            return self.compute_true_likelihood(params)
        
        def log_posterior(params):
            return log_prior(params) + log_likelihood(params)
        
        def proposal(params):
            return np.random.normal(params, 0.1)
        
        def metropolis_hastings(params, proposal):
            new_params = proposal(params)
            log_alpha = log_posterior(new_params) - log_posterior(params)
            if np.log(np.random.uniform()) < log_alpha:
                return new_params
            return params
        

if __name__ == '__main__':
    from spectralcomponents import PowerLaw, Spectrum
    from simulators import Simulator
    from sbi.utils.torchutils import BoxUniform
    from torch import tensor
    from matplotlib import pyplot as plt

    c1 = PowerLaw()
    spectrum = Spectrum(c1)
    params = (0.5, 1.5)
    prior = BoxUniform(low=tensor([0.0, 0.0]), high=tensor([2, 2]))
    simulate = Simulator(spectrum, 10000, pileup='channels', alpha=0.5)
    data = simulate(tensor(params))
    #data = np.ones(1) *600
    print(data)
    self = TruePosterior(prior, spectrum, data, pileup='channels')

    

    # Generate grids for parameters
    # alpha_grid = np.linspace(0.05, 2, 30)
    # beta_grid = np.linspace(0.05, 2, 30)

    # # Compute the posterior over the grid
    # out = self.compute_grid_posterior(alpha_grid, beta_grid).T
    # #out = (alpha_grid[:, np.newaxis] * np.ones_like(beta_grid)[np.newaxis, :]).T
    # # Plotting
    # fig, ax = plt.subplots()
    # cax = ax.imshow(out, extent=(alpha_grid.min(), alpha_grid.max(), beta_grid.min(), beta_grid.max()), origin='lower')
    # ax.set_xlabel('Alpha')
    # ax.set_ylabel('Beta')

    # # Add a colour bar
    # fig.colorbar(cax, ax=ax, label='Posterior Probability')

    # # Mark the true parameters
    # true_alpha, true_beta = params
    # # Convert parameter values to indices
    # alpha_idx = np.argmin(np.abs(alpha_grid - true_alpha))
    # beta_idx = np.argmin(np.abs(beta_grid - true_beta))
    # ax.plot(alpha_grid[alpha_idx], beta_grid[alpha_idx],  'ro')

    # # Show the plot
    # plt.show()