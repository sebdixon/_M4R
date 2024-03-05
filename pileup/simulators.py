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
        rate = self.spectrum.get_rate(*params)
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
        data = np.zeros(self.time_steps)
        total_rate = np.sum(rate)
        rate /= total_rate
        photon_counts = np.random.poisson(total_rate, self.time_steps)
        for time_step, photon_count in enumerate(photon_counts):
            bin_indices = np.nonzero(np.random.multinomial(photon_count, rate))[0] + 1
            total_bin = np.sum(bin_indices) - 1
            if total_bin >= TOTAL_BINS or total_bin < 0:
                continue
            data_t = np.random.multinomial(1, RMF[total_bin])
            data[time_step] = np.nonzero(data_t)[0]
        return data


    def _simulate_channel_pileup1(self, rate):
        data = np.zeros(self.time_steps)
        total_rate = np.sum(rate)
        rate /= total_rate
        photon_counts = np.random.poisson(total_rate, self.time_steps)
        for time_step, photon_count in enumerate(photon_counts):
            if photon_count == 0:
                continue
            bin_indices = np.nonzero(np.random.multinomial(photon_count, rate))[0]
            total_channel = -1  # channel[i] corresponds to bin i
            for bin_index in bin_indices:
                channel = np.nonzero(np.random.multinomial(1, RMF[bin_index]))[0]
                total_channel += channel[0] + 1  # 1 indexing
            if total_channel < TOTAL_CHANNELS:
                data[time_step] = total_channel + 1
        return data
    
    def _simulate_channel_pileup(self, rate):
        channel_indices = np.random.poisson(rate @ RMF, (self.time_steps, TOTAL_CHANNELS)) @ np.arange(1, TOTAL_CHANNELS + 1)
        channel_indices[channel_indices > TOTAL_CHANNELS] = 1
        return channel_indices - 1

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from time import time
    c1 = PowerLaw()
    c1args = (0.05, 1)
    c2 = GaussianEmissionLine()
    c2args = (0.0, 5, 0.05)
    spectrum = Spectrum(c1, c2)
    params = (c1args, c2args)
    start = time()
    simulator = Simulator(spectrum, 10000, pileup='channels')  
    data = simulator(params)
    print (time() - start)
    plt.hist(data[data>0], density=True, bins=40)
    rate = np.concatenate((np.zeros(30), spectrum.get_rate(*params))) @ RMF
    plt.plot(rate / np.sum(rate))