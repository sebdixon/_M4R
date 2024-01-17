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
    data = []
    rate = spectrum.get_rate(*params) @ RMF
    for _ in range(time_steps):
        channels = np.random.poisson(rate) # channel represents count in each channel
        total_channel = int(channels[channels >= 1] @ np.argwhere(channels >= 1))
        if total_channel < TOTAL_CHANNELS:
            data.append(total_channel)
    return data
    

def simulator_var(spectrum: Spectrum, time_steps: int, params: tuple):
    """
    Pileup happens in channel space. Need to apply RMF to each
    photon, then add indices to get pileup.
    """
    data = []
    rate = spectrum.get_rate(*params)
    # This loop is realllyyyyy slow
    for _ in range(time_steps):
        # Need to get bin count first
        bin_count = np.random.poisson(rate)
        total_channel = 0
        for bin_index, photon_count in enumerate(bin_count):
            # For each recorded photon apply RMF
            if photon_count >= 1:
                channels = np.random.multinomial(photon_count, RMF[bin_index])
                total_channel += (np.sum(np.argwhere(channels >= 1)))
        if total_channel < TOTAL_CHANNELS:
            data.append(total_channel)
    return data


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    c1 = PowerLaw()
    c1args = (0.2, 1)
    c2 = GaussianEmissionLine()
    c2args = (0.01, 5, 0.05)
    spectrum = Spectrum(c1, c2)
    params = (c1args, c2args)
    plt.hist(simulator(spectrum, 10000, params), bins=20, density=True)
    rate = spectrum.get_rate(*params) @ RMF
    plt.plot(rate / np.sum(rate))
