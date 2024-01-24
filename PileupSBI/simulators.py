import numpy as np

from spectralcomponents import Spectrum, PowerLaw, GaussianEmissionLine
from txt_inputs.inputs import RMF, ARF, ENERGY_BINS

ARF = np.concatenate((np.zeros(30), ARF))

 # Prob photon arriving in bin j recorded in channel i
RMF = (RMF.T / RMF.sum(axis=1)).T # Need to normalise RMF

RMF = np.vstack((np.zeros((30, 1024)), RMF))

TOTAL_CHANNELS = 1024

def simulator(spectrum: Spectrum, time_steps: int, params: tuple, pileup='bins'):
    """
    Base simulator function for photon incidence.
    """
    rate = spectrum.get_rate(*params)
    if pileup == 'bins':
        return _simulate_bin_pileup(rate, time_steps)
    elif pileup == 'channels':
        return _simulate_channel_pileup(rate, time_steps)
    else:
        raise ValueError('pileup must be one of bins, channels')


def _simulate_bin_pileup(rate, time_steps):
    """
    Pileup happens in bin space. Can just dot rate with RMF.
    """
    rate = rate @ RMF
    data = np.zeros(TOTAL_CHANNELS)
    for _ in range(time_steps):
        channels = np.random.poisson(rate) # channel represents count in each channel
        total_channel = int(channels[channels >= 1] @ np.argwhere(channels >= 1))
        if 0 < total_channel < TOTAL_CHANNELS:
            data[total_channel] += 1
    return data


def _simulate_channel_pileup(rate, time_steps):
    """
    Pileup happens in channel space. Need to apply RMF to each
    photon, then add indices to get pileup.
    """
    data = np.zeros(TOTAL_CHANNELS)
    total_rate = np.sum(rate)
    rate /= total_rate
    photon_counts = np.random.poisson(total_rate, time_steps)
    for photon_count in photon_counts:
        bin_indices = np.nonzero(np.random.multinomial(photon_count, rate))[0]
        total_channel = -1 # channel[i] corresponds to bin i
        for bin_index in bin_indices:
            channel = np.nonzero(np.random.multinomial(1, RMF[bin_index]))[0]
            total_channel += channel[0] + 1 # 1 indexing
        if 0 < total_channel < TOTAL_CHANNELS:
            data[total_channel] += 1
    return data


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from time import time
    c1 = PowerLaw()
    c1args = (0.5, 1)
    c2 = GaussianEmissionLine()
    c2args = (0.1, 5, 0.05)
    spectrum = Spectrum(c1, c2)
    params = (c1args, c2args)
    start = time()
    data = simulator(spectrum, 1000, params, pileup='bins')  
    print (time() - start)
    plt.plot(data / np.sum(data))
    rate = spectrum.get_rate(*params) @ RMF
    plt.plot(rate / np.sum(rate))


