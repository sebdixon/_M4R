import numpy as np

from spectralcomponents import Spectrum, PowerLaw, GaussianEmissionLine
from inputs import RMF, ARF, ENERGY_BINS

# Prob photon arriving in bin j recorded in channel i
class Simulator:
    def __init__(
            self, 
            spectrum, 
            time_steps, 
            pileup='bins', 
            in_RMF=None, 
            in_ARF=None, 
            in_ENERGY_BINS=None,
            alpha=None
    ):
        self.spectrum = spectrum
        self.time_steps = time_steps
        self.pileup = pileup
        self._RMF = in_RMF if in_RMF is not None else RMF
        self._ARF = in_ARF if in_ARF is not None else ARF
        self._ENERGY_BINS = in_ENERGY_BINS if in_ENERGY_BINS is not None else ENERGY_BINS
        self._RMF = np.vstack((np.zeros((30, 1024)), self._RMF))
        self._ARF = np.concatenate((np.zeros(30), self._ARF))
        self.TOTAL_BINS, self.TOTAL_CHANNELS = self._RMF.shape
        self.grade_migration_param = alpha or 0.5

    def __call__(self, params: np.ndarray) -> np.ndarray:
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


    def _simulate_bin_pileup(self, rate: np.ndarray) -> np.ndarray:
        """
        Simulate pileup in bin space.
        """
        data = np.zeros(self.time_steps)
        total_rate = np.sum(rate)
        rate /= total_rate
        photon_counts = np.random.poisson(total_rate, self.time_steps)
        for time_step, photon_count in enumerate(photon_counts):
            bin_indices = np.nonzero(np.random.multinomial(photon_count, rate))[0] + 1
            total_bin = np.sum(bin_indices)
            if total_bin >= self.TOTAL_BINS or total_bin == 0:
                continue
            data_t = np.random.multinomial(1, self._RMF[total_bin])
            data[time_step] = np.nonzero(data_t)[0]
        return data

    
    def _simulate_channel_pileup(self, rate: np.ndarray) -> np.ndarray:
        # records 0 if no photon recorded / photons sum to greater than total
        channel_rate = rate @ self._RMF
        channel_indices = np.random.poisson(
            channel_rate, (self.time_steps, self.TOTAL_CHANNELS))
        Nt = np.sum(channel_indices, axis=1)
        # alpha ^ (Nt-1) is the probability that Nt photons arriving in a time step are recorded as an event
        # in the same time step
        prob_of_no_event = np.random.rand(self.time_steps) > self.grade_migration_param ** (Nt - 1)
        total_channel = channel_indices @ np.arange(1, self.TOTAL_CHANNELS + 1)
        total_channel[prob_of_no_event] = 0
        total_channel[total_channel > self.TOTAL_CHANNELS] = 0 #
        return total_channel


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from time import time
    c1 = PowerLaw()
    c1args = (0.2, 1)
    spectrum = Spectrum(c1)
    params = c1args
    start = time()
    simulator = Simulator(spectrum, 1000, pileup='channels')  
    data = simulator(params)
    print (time() - start)
    plt.hist(data[data>0], density=True, bins=40)
    rate = np.concatenate((np.zeros(30), spectrum.get_rate(params)))
    plt.plot(rate / np.sum(rate))