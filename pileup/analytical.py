import numpy as np
import torch
import numpy as np


from tqdm import tqdm
from scipy.signal import fftconvolve
from scipy.optimize import minimize
from copy import copy

from txt_inputs.inputs import RMF
from utils.pdfs import _poisson_pdf, _normalise, _poisson_inverse_cdf
from spectralcomponents import Spectrum, PowerLaw


def true_likelihood_bins(
        rate: np.ndarray, 
        alpha: float = 0.5
) -> np.ndarray:
    """
    Computes the true likelihood for a model in which pileup happens
    in bin space. This function calculates the likelihood based on the
    given rate array, representing the intensity of the source in each channel.

    Parameters:
    - rate: np.ndarray, the rate (lambda) for each channel.

    Returns:
    - v: np.ndarray, the calculated true likelihood values.
    """
    raise NotImplementedError
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


def true_likelihood_channels(
        rate: np.ndarray, 
        alpha: float = 0.5
) -> np.ndarray:
    """
    Computes the true likelihood for a model in which pileup happens
    in channel space. This function calculates the likelihood based on the
    given rate array, representing the intensity of the source in each channel.

    Parameters:
    - rate: np.ndarray, the rate (lambda) for each channel.
    - alpha: float, the grade migration parameter.

    Returns:
    - v: np.ndarray, the calculated true likelihood values.
    """
    rate = rate @ RMF
    total_rate = np.sum(rate)

    # determine the maximum number of events to consider based on the Poisson distribution
    max_n = _poisson_inverse_cdf(total_rate, 0.0001)
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
        lam_tild_conv[n, :current_result_size] = fftconvolve(
            lam_tild_conv[n - 1, :m + (m - 1) * (n - 1)], 
            lam_tild, 
            mode='full'
        )
        lam_tild_conv[n, :n] = 0

    # extract the valid portion of the convolution
    valid_convolution = lam_tild_conv[:, :m]
    alphas = alpha ** np.arange(max_n + 1)
    v = np.sum(valid_convolution.T * (p_Nt[1:] * alphas), axis=1)
    v0 = 1 - np.sum(v)
    v = np.concatenate(([v0], v))

    return v.astype(np.float64)


def tensor_to_tuple(tensor):
    return tuple(copy(tensor).detach().cpu().numpy())


def tuple_to_tensor(tup):
    return torch.tensor(tup)


class TruePosterior:
    def __init__(
            self, 
            prior,
            spectrum: Spectrum, 
            obs: np.ndarray, 
            pileup: str = 'channels',
    ):
        """
        True posterior class for the pileup model.
        Parameters:
        - prior: torch.distribution, the prior distribution over the parameters.
        - spectrum: Spectrum, the spectrum model.
        - obs: np.ndarray, the observed data.
        - pileup: str, the pileup model to use.
        """
        self.prior = prior
        self.spectrum = spectrum
        self.obs = obs
        self.pileup = pileup


    def compute_log_likelihood(self, params: np.ndarray | torch.Tensor) -> float:
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


    def compute_log_posterior(self, params: np.ndarray | torch.Tensor) -> float:
        """
        Compute the posterior of the parameters given observations x0.
        """
        log_prior = self.prior.log_prob(tuple_to_tensor(params))
        log_likelihood = self.compute_log_likelihood(params)
        return log_likelihood + log_prior


    def compute_grid_posterior(
            self, 
            params1: np.ndarray,
            params2: np.ndarray
    ):
        """
        Compute the posterior of the parameters given observations x0.
        """
        posterior = np.zeros((len(params1), len(params2)))
        for i, param1 in enumerate(tqdm(
            params1, 
            desc='Computing posterior', 
            leave=False)):
            for j, param2 in enumerate(params2):
                posterior[i, j] = self.compute_log_posterior((param1, param2))
        # now normalise so that np.exp(posterior) will sum to 1
        posterior = np.exp(posterior - np.max(posterior))
        return posterior / np.sum(posterior)


    def find_maximum_likelihood(
            self, 
            initial_params: np.ndarray, 
            bounds: np.ndarray
    ) -> np.ndarray:
        """
        Find the maximum likelihood parameters given observations x0.
        """
        result = minimize(
            lambda theta: -self.compute_true_likelihood(theta), 
            initial_params, 
            method='L-BFGS-B')
        self.maximum_likelihood = result.x
        return result.x


def adaptive_metropolis_hastings(
        posterior: TruePosterior, 
        initial_params: np.ndarray, 
        n_samples: int, 
        adapt_for: int = 1000, 
        initial_width: float = 0.1,
        target_accept_rate: float = 0.234
) -> np.ndarray:
    """
    Adaptive Metropolis-Hastings algorithm for sampling from the posterior.

    Parameters:
    - posterior: TruePosterior, the true posterior object.
    - initial_params: np.ndarray, the initial parameters to start from.
    - n_samples: int, the number of samples to generate.
    - adapt_for: int, the number of samples to adapt for.
    - initial_width: float, the initial width of the proposal distribution.
    - target_accept_rate: float, the target acceptance rate.

    Returns:
    - samples: np.ndarray, the samples from the posterior.
    """
    samples = []
    current_params = initial_params
    current_posterior = posterior.compute_log_posterior(current_params)
    proposal_width = initial_width
    accept_count = 0
    
    # Adaptive phase
    pbar = tqdm(range(adapt_for), desc='Adaptive phase')
    for i in pbar:
        proposed_params = current_params + np.random.normal(0, proposal_width, len(initial_params))
        proposed_posterior = posterior.compute_log_posterior(proposed_params)
        
        acceptance_prob = min(1, np.exp(proposed_posterior - current_posterior))
        
        if np.random.rand() < acceptance_prob:
            current_params = proposed_params
            current_posterior = proposed_posterior
            accept_count += 1

        acceptance_rate = accept_count / (i + 1)
        proposal_width *= (1 + 0.1 * (acceptance_rate - target_accept_rate))
        pbar.set_description(f'Adaptive phase - Accept rate: {acceptance_rate:.2f}, Prop width: {proposal_width:.2f}')
    print (f'Proposal width: {proposal_width}')
    accept_count = 0
    
    pbar = tqdm(total=n_samples, desc='Non-adaptive phase')

    while len(samples) < n_samples:
        proposed_params = current_params + np.random.normal(0, proposal_width, len(initial_params))
        proposed_posterior = posterior.compute_log_posterior(proposed_params)
        
        acceptance_prob = min(1, np.exp(proposed_posterior - current_posterior))
        
        if np.random.rand() < acceptance_prob:
            current_params = proposed_params
            current_posterior = proposed_posterior
            accept_count += 1
            pbar.update(1)
            samples.append(current_params)
    
    return np.array(samples)


def adaptive_metropolis_hastings2(
        posterior: TruePosterior, 
        initial_params: np.ndarray, 
        n_samples: int, 
        adapt_for: int = 1000, 
        initial_cov: np.ndarray = None,
        target_accept_rate: float = 0.234,
        epsilon: float = 1e-6,
        min_accept_rate: float = 0.1,
        max_accept_rate: float = 0.5
) -> np.ndarray:
    """
    Adaptive Metropolis-Hastings algorithm for sampling from the posterior.

    Parameters:
    - posterior: TruePosterior, the true posterior object.
    - initial_params: np.ndarray, the initial parameters to start from.
    - n_samples: int, the number of samples to generate.
    - adapt_for: int, the number of samples to adapt for.
    - initial_cov: np.ndarray, the initial covariance matrix of the proposal distribution.
    - target_accept_rate: float, the target acceptance rate.
    - epsilon: float, regularization term to ensure numerical stability.
    - min_accept_rate: float, minimum acceptable acceptance rate.
    - max_accept_rate: float, maximum acceptable acceptance rate.

    Returns:
    - samples: np.ndarray, the samples from the posterior.
    """
    if initial_cov is None:
        initial_cov = np.eye(len(initial_params)) * 0.01

    samples = []
    current_params = initial_params
    current_posterior = posterior.compute_log_posterior(current_params)
    cov_matrix = initial_cov
    mean_params = np.zeros_like(initial_params)
    accept_count = 0
    d = len(initial_params)
    s = 2.38**2 / d

    # Adaptive phase
    pbar = tqdm(range(adapt_for), desc='Adaptive phase')
    for i in pbar:
        proposed_params = np.random.multivariate_normal(current_params, cov_matrix)
        proposed_posterior = posterior.compute_log_posterior(proposed_params)
        
        acceptance_prob = min(1, np.exp(proposed_posterior - current_posterior))
        print (acceptance_prob)
        if np.random.rand() < acceptance_prob:
            current_params = proposed_params
            current_posterior = proposed_posterior
            accept_count += 1

        samples.append(current_params)
        acceptance_rate = accept_count / (i + 1)
        
        # Update mean
        old_mean_params = mean_params.copy()
        mean_params = (mean_params * i + current_params) / (i + 1)
        
        # Update covariance matrix
        if i == 0:
            cov_matrix = np.outer(current_params - mean_params, current_params - mean_params)
        else:
            cov_matrix = ((i - 1) * cov_matrix + (i / (i + 1)) * np.outer(current_params - mean_params, current_params - old_mean_params)) / i
        
        # Regularize and scale covariance matrix
        cov_matrix = s * (cov_matrix + epsilon * np.eye(d))
        
        # Ensure covariance matrix does not become excessively large
        if np.linalg.det(cov_matrix) > 1e2:
            cov_matrix = np.eye(d) * 0.01
        
        # Adjust adaptation rate
        if acceptance_rate < min_accept_rate:
            s *= 0.9
        elif acceptance_rate > max_accept_rate:
            s *= 1.1
        
        pbar.set_description(f'Adaptive phase - Accept rate: {acceptance_rate:.2f}, Scale: {s:.2f}')

    print(f'Final covariance matrix: {cov_matrix}')
    accept_count = 0
    pbar = tqdm(total=n_samples, desc='Non-adaptive phase')

    while len(samples) < n_samples:
        proposed_params = np.random.multivariate_normal(current_params, cov_matrix)
        proposed_posterior = posterior.compute_log_posterior(proposed_params)
        
        acceptance_prob = min(1, np.exp(proposed_posterior - current_posterior))
        
        if np.random.rand() < acceptance_prob:
            current_params = proposed_params
            current_posterior = proposed_posterior
            accept_count += 1
            pbar.update(1)
            samples.append(current_params)
    
    return np.array(samples)

def get_true_posterior_samples_power_law(
        data_filepath: str ='simulated_data/power_law/x0_power_law.npy',
        save_filepath: str ='simulated_data/power_law/posterior_samples_AMHMCMC.npy',
        prior: torch.distributions.distribution = None,
        spectrum: Spectrum = None,
        pileup: str = 'channels',
        initial_params: np.ndarray | torch.Tensor = None,
        n_samples: int = 1000,
        adapt_for: int = 1000,
        initial_width: float = 0.1,
        target_accept_rate: float = 0.234,
        total_observed: int = 10000
) -> np.ndarray:
    """
    Samples from the true posterior of the power law model using the adaptive
    Metropolis-Hastings algorithm.

    DATA MUST BE IN TIMESERIES FORM.

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
        prior = BoxUniform(low=torch.tensor([0.1, 0.1]), high=torch.tensor([2, 2]))
    if spectrum is None:
        c1 = PowerLaw()
        spectrum = Spectrum(c1)
    if initial_params is None:
        initial_params = np.array([0.5, 1.5])
    obs = np.load(data_filepath)
    data = channels_to_timeseries(obs, total_observed)
    posterior = TruePosterior(prior, spectrum, data, pileup)
    samples = adaptive_metropolis_hastings(
        posterior, 
        initial_params, 
        n_samples, 
        adapt_for, 
        initial_width, 
        target_accept_rate)
    save_true_posterior_samples(save_filepath, samples)
    return samples


def save_true_posterior_samples(save_filepath: str, samples: np.ndarray):
    """
    Shuffles samples and saves to 10 chunks
    """
    np.random.shuffle(samples)
    chunk_size = len(samples) // 10
    for i in range(10):
        chunk = samples[i * chunk_size:(i + 1) * chunk_size]
        np.save(save_filepath.replace('.npy', f'_{i}.npy'), chunk)
            


def channels_to_timeseries(channels: np.ndarray, total_events: int) -> np.ndarray:
    """
    Convert a channels representation to a timeseries representation.

    Parameters:
    - channels: np.ndarray, the channels representation (counts per channel).
    - total_events: int, the total number of events to be included in the timeseries.

    Returns:
    - timeseries: np.ndarray, the timeseries representation.
    """
    timeseries = np.zeros(total_events, dtype=int)
    index = 0
    for i, count in enumerate(channels):
        timeseries[index:index + count] = i
        index += count
    return timeseries


if __name__ == '__main__':
    from spectralcomponents import PowerLaw, Spectrum
    from simulators import Simulator
    from sbi.utils.torchutils import BoxUniform
    from torch import tensor
    from matplotlib import pyplot as plt

    c1 = PowerLaw()
    spectrum = Spectrum(c1)
    params = (0.5, 1.5)
    prior = BoxUniform(low=tensor([0.1, 0.1]), high=tensor([2, 2]))
    simulate = Simulator(spectrum, 10000, pileup='channels', alpha=0.5)

    data = channels_to_timeseries(
        np.load('simulated_data/power_law/x0_power_law.npy'),
        10000
    )

    # adaptive_metropolis_hastings2(
    #     posterior=TruePosterior(prior, spectrum, data, 'channels'),
    #     initial_params=params,
    #     n_samples=10000,
    #     adapt_for=1000,
    #     initial_cov=None,
    # )


    get_true_posterior_samples_power_law(
        data_filepath='simulated_data/power_law/x0_power_law.npy',
        save_filepath='simulated_data/power_law/posterior_samples_AMHMCMC.npy',
        prior=prior,
        spectrum=spectrum,
        pileup='channels',
        initial_params=params,
        n_samples=10000,
        adapt_for=1000,
        initial_width=0.2,
        target_accept_rate=0.234
    )