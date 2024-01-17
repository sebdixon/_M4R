import numpy as np
from numba import njit
from spectralcomponents import ARF, E_BAR, Spectrum, PowerLaw, GaussianEmissionLine

ARF = np.concatenate((np.zeros(30), ARF))

RMF = np.loadtxt('txt_inputs/rmf.txt') # Prob photon arriving in bin j recorded in channel i
RMF = (RMF.T / RMF.sum(axis=1)).T # Need to normalise RMF

RMF = np.vstack((np.zeros((30, 1024)), RMF))

TOTAL_CHANNELS = 1024

def simulator(spectrum: Spectrum, time_steps: int, params: tuple):
    """
    Pileup happens in bin space. Can just dot rate with RMF.
    """
    data = np.zeros(TOTAL_CHANNELS)
    rate = spectrum.get_rate(*params) @ RMF
    for _ in range(time_steps):
        channels = np.random.poisson(rate) # channel represents count in each channel
        total_channel = int(channels[channels >= 1] @ np.argwhere(channels >= 1))
        if total_channel < TOTAL_CHANNELS:
            data[total_channel] += 1
    return data
    

def simulator_var(spectrum: Spectrum, time_steps: int, params: tuple):
    """
    Pileup happens in channel space. Need to apply RMF to each
    photon, then add indices to get pileup.
    """
    rate = spectrum.get_rate(*params)
    return _simulate_channel_pileup(rate, time_steps)


def _simulate_channel_pileup1(rate:np.ndarray, time_steps:int):
    data = np.zeros(TOTAL_CHANNELS)
    for _ in range(time_steps):
        # Need to get bin count first
        bin_count = np.random.poisson(rate)
        total_channel = 0
        for bin_index, photon_count in enumerate(bin_count):
            # For each recorded photon apply RMF
            if photon_count >= 1:
                channels = np.random.multinomial(photon_count, RMF[bin_index])
                # Thinking about the line below - if u had 2 photon in channel 1 (index 0)
                # then u would want the total to be channel 2 (index 1).
                # If u had 3 in channel 1 u would want channel 3
                # Generally it would be the sum of the indices of channels plus
                # the number of photons - 1
                total_channel += (np.sum(np.argwhere(channels >= 1) + 1)) - 1
        if 0 < total_channel < TOTAL_CHANNELS:
            data[total_channel] += 1
    return data


def _simulate_channel_pileup(rate, time_steps):
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
    c1args = (0.2, 1)
    c2 = GaussianEmissionLine()
    c2args = (0.1, 5, 0.05)
    spectrum = Spectrum(c1, c2)
    params = (c1args, c2args)
    start = time()
    data = simulator_var(spectrum, 1000, params)  
    print (time() - start)
    plt.plot(data / np.sum(data))
    rate = spectrum.get_rate(*params) @ RMF
    plt.plot(rate / np.sum(rate))




