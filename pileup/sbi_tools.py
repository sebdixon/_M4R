import numpy as np
import torch
from sbi.inference import SNPE, SNRE_C
from sbi.utils import posterior_nn
from torch import Tensor
from torch.distributions import Normal, MultivariateNormal, Uniform, Independent
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.inference import SNRE_B


# Useful priors
class TruncatedNormal(MultivariateNormal):
    # Not yet rely working
    def __init__(self, loc, covariance, low, high):
        super().__init__(loc, covariance)
        self.low = low
        self.high = high
    
    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        
        samples = torch.empty(sample_shape + self._batch_shape).to(self.loc.device)
        mask = torch.ones_like(samples, dtype=torch.bool)
        
        while mask.any():
            new_samples = super().sample(sample_shape)
            within_bounds = (new_samples >= self.low) & (new_samples <= self.high)
            samples = torch.where(within_bounds, new_samples, samples)
            mask &= ~within_bounds
            
        return samples


class BoxUniform(Independent):
    def __init__(self, low: Tensor, high: Tensor, reinterpreted_batch_ndims: int = 1, device='cpu'):
        super().__init__(
                Uniform(
                    low=torch.as_tensor(
                        low, dtype=torch.float32, device=torch.device(device)
                    ),
                    high=torch.as_tensor(
                        high, dtype=torch.float32, device=torch.device(device)
                    ),
                    validate_args=False,
                ),
                reinterpreted_batch_ndims,
            )


class SymmetricTruncatedNormal(Normal):
    def __init__(self, loc, scale):
        super().__init__(loc, scale)
    
    def sample(self, sample_shape=torch.Size()):
        return torch.abs(super().sample(sample_shape))


def get_SNPE_posterior(prior, simulator, embedding_net, theta=None, x=None, **args):
    simulator, prior = prepare_for_sbi(simulator, prior)
    neural_posterior = posterior_nn(
    model="maf", embedding_net=embedding_net, hidden_features=10, num_transforms=2)
    inference = SNPE(prior=prior, density_estimator=neural_posterior)
    if theta is None or x is None:
        theta, x = simulate_for_sbi(simulator, proposal=prior, **args)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    return posterior


def get_SNRE_posterior(prior, simulator, theta=None, x=None, **args):
    simulator, prior = prepare_for_sbi(simulator, prior)
    inference = SNRE_B(prior)
    if theta is None or x is None:
        theta, x = simulate_for_sbi(simulator, proposal=prior, **args)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    return posterior