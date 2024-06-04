import numpy as np
import torch
from sbi import utils as utils
from sbi import analysis as analysis
from simulators import Simulator
from spectralcomponents import PowerLaw, GaussianEmissionLine, Spectrum
from sbi.inference import prepare_for_sbi, simulate_for_sbi
from sbi_tools import BoxUniform


def simulate_simple(
        spectrum: Spectrum, 
        params: np.ndarray, 
        n_frames: int = 10000
) -> np.ndarray:
    simulator =  Simulator(spectrum, n_frames, pileup='channels')
    data = simulator(params)
    return np.bincount(data, minlength=1025)[1:]


def _simulate_simple(params: np.ndarray) -> np.ndarray:
    """Allows to be passed into inference."""    
    return simulate_simple(spectrum, params)


def save_simulations_in_chunks_power_law(
        prior, 
        start: int = 1,
        chunks: int = 10, 
        simulations_per_chunk: int = 1000
) -> None:
    simulator, prior = prepare_for_sbi(_simulate_simple, prior)
    for chunk in range(start, chunks+1):
        theta, x = simulate_for_sbi(
            simulator, 
            prior, 
            num_simulations=simulations_per_chunk
        )
        np.save(f'simulated_data/power_law/x_chunk{chunk}_power_law.npy', x)
        np.save(f'simulated_data/power_law/theta_chunk{chunk}_power_law.npy', theta)


if __name__ == '__main__':
    c1 = PowerLaw()
    spectrum = Spectrum(c1)
    prior = BoxUniform(low=torch.tensor([0.1, 0.1]), high=torch.tensor([2, 2]))
    save_simulations_in_chunks_power_law(
        prior,
        start=11,
        chunks=100,)