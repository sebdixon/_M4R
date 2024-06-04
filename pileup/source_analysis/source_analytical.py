import os
import sys
import numpy as np

parent_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from analytical import TruePosterior, adaptive_metropolis_hastings, channels_to_timeseries
from source_ciao import XspecModel

X0 = np.loadtxt('xspec_inputs/counts.txt')

class XspecPrior:
    def __init__(self):
        self.low = np.array([0.1, 0.1])
        self.high = np.array([50, 50])

    def sample(self, n):
        return np.random.uniform(
            low=self.low, 
            high=self.high, 
            size=(n, 2)
        )

    def log_prob(self, params):
        kT, norm = params
        if kT < self.low[0] or kT > self.high[0]:
            return -np.inf
        if norm < self.low[1] or norm > self.high[1]:
            return -np.inf
        return -np.log((self.high[0] - self.low[0])) + \
               -np.log((self.high[1] - self.low[1]))

def get_true_posterior_samples_vapec(
        data: np.ndarray = X0.astype(int),
        save_filepath: str ='xspec_outputs/ciao_posterior_samples_AMHMCMC.npy',
        prior: XspecPrior = XspecPrior(),
        spectrum: XspecModel = XspecModel(),
        pileup: str = 'channels',
        representation: str = 'channels',
        initial_params: np.ndarray = None,
        n_samples: int = 10000,
        adapt_for: int = 1000,
        initial_width: float = 0.1,
        target_accept_rate: float = 0.234
) -> np.ndarray:
    """
    Samples from the true posterior of the power law model using the adaptive
    Metropolis-Hastings algorithm.

    Parameters:
    - data_filepath: str, the filepath to the observed data.
    - save_filepath: str, the filepath to save the samples.
    - prior: torch.distributions.distribution, the prior distribution over the parameters.
    - spectrum: Spectrum, the spectrum model.
    - pileup: str, the pileup model to use.
    - initial_params: np.ndarray, the initial parameters to start from.
    - n_samples: int, the number of samples to generate.
    - adapt_for: int, the number of samples to adapt for.
    - initial_width: float, the initial width of the proposal distribution.
    - target_accept_rate: float, the target acceptance rate.

    Returns:
    - samples: np.ndarray, the samples from the posterior.
    """
    if prior is None:
        prior = XspecPrior()
    if spectrum is None:
        spectrum = XspecModel()
    if initial_params is None:
        initial_params = np.array([1.6, 3.8])
    obs = channels_to_timeseries(data, 14479)
    posterior = TruePosterior(prior, spectrum, obs, pileup)
    samples = adaptive_metropolis_hastings(
        posterior, 
        initial_params, 
        n_samples, 
        adapt_for, 
        initial_width, 
        target_accept_rate)
    np.save(save_filepath, samples)
    return samples

if __name__ == '__main__':
    samples = get_true_posterior_samples_vapec()
    print('done')