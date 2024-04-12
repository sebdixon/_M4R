from copy import copy
import numpy as np
import torch
from tqdm import tqdm

from scipy.signal import fftconvolve

from inputs import RMF
from utils.pdfs import _poisson_pdf, _normalise, _poisson_inverse_cdf


def true_likelihood1(rate: np.ndarray):
    """
    Computes the true likelihood for a model in which pileup happens
    in bin space. This function calculates the likelihood based on the
    given rate array, representing the intensity of the source in each channel.

    Parameters:
    - rate: np.ndarray, the rate (lambda) for each channel.

    Returns:
    - v: np.ndarray, the calculated true likelihood values.
    """
    rate = rate @ RMF
    total_rate = np.sum(rate)

    # determine the maximum number of events to consider based on the Poisson distribution
    max_n = _poisson_inverse_cdf(total_rate, 0.01)

    # calculate the Poisson distribution PDF for required ns
    p_Nt = _poisson_pdf(total_rate, np.arange(max_n))
    p_Nt = _normalise(p_Nt)  # normalise the Poisson PDF

    lam_tild = np.concatenate((np.zeros(30), rate / total_rate))
    m = len(lam_tild)

    lam_tild_conv = np.zeros((max_n, m))
    lam_tild_conv[0, :] = lam_tild

    # convolution iteratively for each number of events
    for n in range(1, max_n):
        lam_tild_conv[n, :] = fftconvolve(lam_tild_conv[n - 1, :m], lam_tild, mode='full')[:m * 2 - 1]
        lam_tild_conv[n, :n] = 0
    v = np.sum(lam_tild_conv.T * p_Nt, axis=1)
    #v[0] += 1 - np.sum(v[1:])
    return v.astype(np.float64)


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
    rate = rate @ RMF
    total_rate = np.sum(rate)

    # determine the maximum number of events to consider based on the Poisson distribution
    max_n = _poisson_inverse_cdf(total_rate, 0.001)

    # calculate the Poisson distribution PDF for required ns
    p_Nt = _poisson_pdf(total_rate, np.arange(max_n + 1))
    p_Nt = _normalise(p_Nt)  # normalise the Poisson PDF

    lam_tild = rate / total_rate
    m = len(lam_tild)

    lam_tild_conv = np.zeros((max_n + 1, m))
    lam_tild_conv[0, :] = lam_tild
    # convolution iteratively for each number of events
    for n in range(1, max_n + 1):
        lam_tild_conv[n, :] = fftconvolve(lam_tild_conv[n - 1, :], lam_tild, mode='same')
    v = np.sum(lam_tild_conv.T * p_Nt, axis=1)
    #v0 = p_Nt[0] + (1 - np.sum(lam_tild_conv, axis=1)) @ p_Nt
    v0 = 1 - np.sum(v) # mthmaticaly equivalent to the above??
    #print (f"pr no ev {prob_of_no_event}")
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
        likelihood = true_likelihood(rate)
        
        log_likelihood_of_x0 = np.sum(np.log(likelihood[x0]))
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
        x0 = self.obs

        posterior = np.zeros((len(params1), len(params2)))
        for i, param1 in enumerate(tqdm(params1, desc='Computing posterior', leave=False)):
            for j, param2 in enumerate(params2):
                posterior[i, j] = self.compute_log_posterior((param1, param2))
        # now normalise so that np.exp(posterior) will sum to 1
        posterior = np.exp(posterior - np.max(posterior))
        return posterior / np.sum(posterior)


if __name__ == '__main__':
    from spectralcomponents import PowerLaw, Spectrum
    from simulators import Simulator
    from sbi.utils.torchutils import BoxUniform
    from torch import tensor

    c1 = PowerLaw()
    spectrum = Spectrum(c1)
    params = (0.2, 0.5)
    prior = BoxUniform(low=tensor([0.0, 0.0]), high=tensor([1, 1]))
    simulate = Simulator(spectrum, 1000, pileup='channels')
    data = simulate(tensor(params))
    print(data)
    self = TruePosterior(prior, spectrum, data)

    # Generate grids for parameters
    alpha_grid = np.linspace(0.0, 2, 200)
    beta_grid = np.linspace(0.0, 2, 200)

    # Compute the posterior over the grid
    out = self.compute_grid_posterior(alpha_grid, beta_grid)

    # Plotting
    fig, ax = plt.subplots()
    cax = ax.imshow(out, extent=(alpha_grid.min(), alpha_grid.max(), beta_grid.min(), beta_grid.max()), origin='lower')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Beta')

    # Add a color bar
    fig.colorbar(cax, ax=ax, label='Posterior Probability')

    # Mark the true parameters
    true_alpha, true_beta = params
    # Convert parameter values to indices
    alpha_idx = np.argmin(np.abs(alpha_grid - true_alpha))
    beta_idx = np.argmin(np.abs(beta_grid - true_beta))
    ax.plot(alpha_grid[alpha_idx], beta_grid[beta_idx], 'ro')

    # Show the plot
    plt.show()