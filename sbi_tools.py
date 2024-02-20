import numpy as np
import torch
from sbi.inference import SNPE, SNRE_C
from sbi.utils import get_density_thresholder, RestrictedPrior
from torch import Tensor
from torch.distributions import Normal, Uniform
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi


# Useful priors
class TruncatedGaussian(Normal):
    def __init__(self, loc, scale, low, high):
        if not isinstance(loc, Tensor):
            loc = Tensor([loc]) 
        if not isinstance(scale, Tensor):   
            scale = Tensor([scale])
        super().__init__(loc, scale)
        self.low = low
        self.high = high
    
    def sample(self, shape=1):
        sample =  super().sample()
        if (sample < self.low).any() or (sample > self.high).any():
            return self.sample()
        return sample
    

def get_SNPE_posterior(prior, simulator):
    simulator, prior = prepare_for_sbi(simulator, prior)
    inference = SNPE(prior)

    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=1000)
    density_estimator = inference.append_simulations(theta, x).train()
    posterior = inference.build_posterior(density_estimator)
    return posterior


def do_contrastive_ratio(prior, simulator, num_sims, x_o):
    # Amortized inference
    inference = SNRE_C(prior)
    proposal = prior
    theta = proposal.sample((num_sims,))
    x = simulator(theta)
    _ = inference.append_simulations(theta, x).train(
        num_classes=5,  # SNRE_C sees `2 * num_classes - 1` marginally drawn contrastive pairs.
        gamma=1.0,  # SNRE_C can control the weight between terms in its loss function.
    )
    posterior = inference.build_posterior().set_default_x(x_o)
    return posterior


if __name__ == '__main__':
    prior = TruncatedGaussian(torch.ones(2), torch.ones(2) * 10, 0, 1)
    
    