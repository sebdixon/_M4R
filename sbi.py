import numpy as np
import sbi
from sbi.inference import SNPE, SNRE_C
from sbi.utils import get_density_thresholder, RestrictedPrior
from torch import Size
from torch._C import Size
from torch.distributions import Normal, Uniform


# Useful priors
class TruncatedGaussian(Normal):
    def __init__(self, loc, scale, low, high):
        super().__init__(loc, scale)
        self.low = low
        self.high = high
    
    def sample(self):
        sample =  super().sample()
        if sample < self.low or sample > self.high:
            return self.sample()
    

def do_SNPE(prior, simulator, rounds, num_sims, x_o):
    inference = SNPE(prior)
    proposal = prior
    for _ in range(rounds):
        theta = proposal.sample((num_sims,))
        x = simulator(theta)
        _ = inference.append_simulations(theta, x).train(force_first_round_loss=True)
        posterior = inference.build_posterior().set_default_x(x_o)

        accept_reject_fn = get_density_thresholder(posterior, quantile=1e-4)
        proposal = RestrictedPrior(prior, accept_reject_fn, sample_with="rejection")
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