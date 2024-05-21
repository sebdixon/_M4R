# conda deactivate
# conda activate ciao
import sys
import os
import numpy as np

from sherpa.astro import xspec
from tqdm import tqdm

parent_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from simulators import Simulator
from txt_inputs.inputs import process_rmf


# globals
if __name__ == '__main__':
    ROUND = int(os.getenv('ROUND'))
    NUM_SIMS = int(os.getenv('NUM_SIMS'))
else:
    ROUND = 1
    NUM_SIMS = 1000

RMF = process_rmf(np.loadtxt('xspec_inputs/rmf.txt'))
ARF = np.loadtxt('xspec_inputs/arf.txt')
X0 = np.loadtxt('xspec_inputs/counts.txt')

T = 14479
DELTA_T = 3.2
DELTA_B = 0.01
E_LO = np.linspace(0.3, 10.99, 1070)
E_HI = np.linspace(0.31, 11.00, 1070)
E_BAR = (E_LO + E_HI) / 2

print ('ciao running round ', ROUND)

class XspecModel:
    def __init__(self):
        self.model = xspec.XSvapec()

    def get_rate(self, params: np.ndarray) -> np.ndarray:
        self.model.kT = params[0]
        self.model.norm = params[1]
        return self.model(E_LO, E_HI) * DELTA_T * \
            DELTA_B * ARF


def simulate_simple(params):
    model = XspecModel()
    simulator =  Simulator(
        spectrum=model,
        time_steps=T,
        pileup='channels',
        in_RMF=RMF,
        in_ARF=ARF,
        in_ENERGY_BINS=E_BAR,
        alpha=0.5)
    data = simulator(params)
    return np.bincount(data, minlength=1025)[1:]


def sample_prior_2d():
    return np.array([np.random.uniform(0.1, 50), 
                     np.random.uniform(0.1, 50)])


def sample_prior_3d():
    return np.array([np.random.uniform(0.1, 50), 
                     np.random.uniform(0.1, 50), 
                     np.random.uniform(0.1, 1.)])


def simulate_ciao(round):
    xs = np.zeros((NUM_SIMS, 1024))
    thetas = np.zeros((NUM_SIMS, 2))
    if round == 1:
        for i in tqdm(range(NUM_SIMS)):
            theta = sample_prior_2d()
            thetas[i] = theta
            xs[i] = simulate_simple(theta)
        np.save(
            f'xspec_outputs/ciao_theta_{round}.npy', thetas)
    elif round >= 2:
        thetas = np.load(
            f'xspec_outputs/ciao_theta_{round}.npy')
        for i in tqdm(range(NUM_SIMS)):
            theta = thetas[i]
            xs[i] = simulate_simple(theta)
    np.save(f'xspec_outputs/ciao_x_{round}.npy', xs)

if __name__ == '__main__':
    simulate_ciao(ROUND)
    print('done')
    