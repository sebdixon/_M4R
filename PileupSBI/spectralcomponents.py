import numpy as np
import torch

# Globals
ARF = np.loadtxt('txt_inputs/arf.txt') # Effective area
#ARF = np.concatenate((np.zeros(30), ARF))
ENERGY_BINS = np.loadtxt('txt_inputs/energy_bins.txt') # Lower bound of bins
#ENERGY_BINS = np.concatenate((np.zeros(30), ENERGY_BINS))

BIN_WIDTH = ENERGY_BINS[1] - ENERGY_BINS[0]
E_BAR = ENERGY_BINS + BIN_WIDTH / 2

# Pre-defined params (edit as needed)
TIME_WIDTH = 3.2

def _normal_pdf(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * \
        np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))


class SpectralComponent():
    """
    Base class from which other spectraL components inherit.
    Stil considering the best way to implement this.
    """
    def __init__(self, *params):
        self.params = params
    def get_rate(self):
        raise NotImplementedError()


class PowerLaw(SpectralComponent):
    """
    Basic power law spectrum. Params alpha + beta.
    """
    def get_rate(self, alpha, beta):
        return alpha * torch.Tensor(E_BAR) ** -beta


class GaussianEmissionLine(SpectralComponent):
    """
    Emision line centred at mu w std sigma. Added power
    param as well bc unsure if should exist or not. Considering
    fixing sigma=0.1.
    """
    def get_rate(self, power, mu, sigma):
        return power * _normal_pdf(E_BAR, mu, sigma)


class Spectrum():
    """
    Spectrum class comprised of spectral components.
    Can pass arbitrarily many spectral components in. Must 
    ensure that params are passed as tuple of tuples in the 
    same order in which the components are passed in.
    """
    def __init__(self, *components):
        self.components = components
    
    def get_rate(self, *params):
        rate = np.zeros_like(E_BAR)
        for component, component_params in zip(self.components, params):
            print (component, component_params)
            rate += component.get_rate(*component_params)
        self.spectrum = rate.copy()
        rate *= TIME_WIDTH * BIN_WIDTH * ARF
        rate = np.concatenate((np.zeros(30), rate))
        return rate


if __name__ == '__main__':
    c1 = PowerLaw()
    c1args = (1,1)
    c2 = GaussianEmissionLine()
    c2args = (0.1, 5, 0.05)
    self = Spectrum(c1, c2)
    from matplotlib import pyplot as plt
    plt.plot(self.get_rate(c1args, c2args))
    """
    plt.plot(self.spectrum)
    plt.show()
    plt.plot(self.rate)

    plt.show()"""