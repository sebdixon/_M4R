import numpy as np
import torch
from sbi.inference import SNPE, SNRE_C
from sbi.utils import get_density_thresholder, RestrictedPrior
from torch import Tensor
from torch.distributions import Normal, MultivariateNormal, Uniform
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi


# Useful priors
class TruncatedNormal(Normal):
    # Not yet rely working
    def __init__(self, loc, scale, low, high):
        super().__init__(loc, scale)
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


class SymmetricTruncatedNormal(Normal):
    def __init__(self, loc, scale):
        super().__init__(loc, scale)
    
    def sample(self, sample_shape=torch.Size()):
        return torch.abs(super().sample(sample_shape))


def get_SNPE_posterior(prior, simulator, **args):
    simulator, prior = prepare_for_sbi(simulator, prior)
    inference = SNPE(prior)
    theta, x = simulate_for_sbi(simulator, proposal=prior, **args)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    return posterior


def do_contrastive_ratio(prior, simulator, num_sims, x_o):
    # Amortized inference
    # This doesn't yet work
    inference = SNRE_C(prior)
    proposal = prior
    theta = proposal.sample((num_sims,))
    x = simulator(theta)
    _ = inference.append_simulations(theta, x).train(
        num_classes=5,
        gamma=1.0,
    )
    posterior = inference.build_posterior().set_default_x(x_o)
    return posterior