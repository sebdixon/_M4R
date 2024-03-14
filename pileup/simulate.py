import torch

from sbi_tools import BoxUniform
from spectralcomponents import PowerLaw
from simulators import Simulator
from utils.data_formats import timeseries_to_channels
from sbi.inference import prepare_for_sbi, simulate_for_sbi

c1 = PowerLaw()
true_params = (0.5, 1)


def simulate_channels(params):
    simulator =  Simulator(c1, pileup='channels', time_steps=10000)
    return timeseries_to_channels(simulator((params)))


def simulate_box_uniform_prior(low, high, filename):
    prior = BoxUniform(low=low, high=high)
    simulator, prior = prepare_for_sbi(simulate_channels, prior)
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=100000)
    torch.save(theta, filename + '_theta.pt')
    torch.save(x, filename + '_x.pt')

if __name__ == '__main__':
    simulate_box_uniform_prior(torch.Tensor([0.1, 0.1]), torch.Tensor([2, 2]), 'data/boxuniform_prior_2d_2')
    print('done')
