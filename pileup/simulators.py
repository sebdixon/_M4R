import numpy as np

from spectralcomponents import Spectrum, PowerLaw, GaussianEmissionLine
from inputs import RMF, ARF, ENERGY_BINS

ARF = np.concatenate((np.zeros(30), ARF))

# Prob photon arriving in bin j recorded in channel i
RMF = np.vstack((np.zeros((30, 1024)), RMF))

TOTAL_BINS, TOTAL_CHANNELS = RMF.shape


class Simulator:
    def __init__(self, spectrum, time_steps, pileup='bins'):
        self.spectrum = spectrum
        self.time_steps = time_steps
        self.pileup = pileup

    def __call__(self, params):
        """
        Call the simulator with a specified pileup type.
        """
        rate = self.spectrum.get_rate(params)
        rate = np.concatenate((np.zeros(30), rate))  # Assuming 30 is a specific requirement

        if self.pileup == 'bins':
            return self._simulate_bin_pileup(rate)
        elif self.pileup == 'channels':
            return self._simulate_channel_pileup(rate)
        else:
            raise ValueError('pileup must be one of "bins", "channels"')


    def _simulate_bin_pileup(self, rate):
        """
        Simulate pileup in bin space.
        """
        data = np.zeros(self.time_steps) - 1
        total_rate = np.sum(rate)
        rate /= total_rate
        photon_counts = np.random.poisson(total_rate, self.time_steps)
        for time_step, photon_count in enumerate(photon_counts):
            bin_indices = np.nonzero(np.random.multinomial(photon_count, rate))[0] + 1
            total_bin = np.sum(bin_indices) - 1
            if total_bin >= TOTAL_BINS:
                continue
            data_t = np.random.multinomial(1, RMF[total_bin])
            data[time_step] = np.nonzero(data_t)[0]
        return data

    
    def _simulate_channel_pileup(self, rate):
        # records 0 if no photon recorded / photons sum to greater than total
        channel_indices = np.random.poisson(rate @ RMF, (self.time_steps, TOTAL_CHANNELS)) @ np.arange(1, TOTAL_CHANNELS + 1)
        channel_indices[channel_indices > TOTAL_CHANNELS] = 0 #
        return channel_indices


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from time import time
    c1 = PowerLaw()
    c1args = (0.2, 1)
    spectrum = Spectrum(c1)
    params = c1args
    start = time()
    simulator = Simulator(spectrum, 100000, pileup='bins')  
    data = simulator(params)
    print (time() - start)
    plt.hist(data, density=True, bins=40)
    rate = np.concatenate((np.zeros(30), spectrum.get_rate(params))) @ RMF
    plt.plot(rate / np.sum(rate))