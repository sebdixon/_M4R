import numpy as np
import torch

from utils.pdfs import _normal_pdf
from inputs import RMF, ARF, ENERGY_BINS
#ENERGY_BINS = np.concatenate((np.zeros(30), ENERGY_BINS))

BIN_WIDTH = ENERGY_BINS[1] - ENERGY_BINS[0]
E_BAR = ENERGY_BINS + BIN_WIDTH / 2

# Pre-defined params (edit as needed)
TIME_WIDTH = 3.2



class SpectralComponent():
    """
    Base class from which other spectraL components inherit.
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
        return np.array(alpha * torch.Tensor(E_BAR) ** -beta)


class GaussianEmissionLine(SpectralComponent):
    """
    Emision line centred at mu w std sigma. Added power
    param as well bc unsure if should exist or not. Considering
    fixing sigma=0.1.
    """
    def get_rate(self, power, mu, sigma):
        return power * _normal_pdf(E_BAR, mu, sigma)


class BrokenPowerLaw(SpectralComponent):
    """
    Broken power law spectrum. Params alpha1, beta1, alpha2, beta2, break energy.
    """
    def get_rate(self, alpha1, beta1, alpha2, beta2, e_break):
        return np.array(alpha1 * torch.Tensor(E_BAR[E_BAR < e_break]) ** -beta1) + np.array(alpha2 * torch.Tensor(E_BAR[E_BAR >= e_break]) ** -beta2)
    

class BlackBody(SpectralComponent):
    """
    Black body spectrum. Params T, normalization.
    """
    def get_rate(self, T, norm):
        return norm * (8 * np.pi * np.array(E_BAR) ** 2) / (np.exp(E_BAR / T) - 1)
    

class Bremsstrahlung(SpectralComponent):
    """
    Bremsstrahlung spectrum. Params T, normalization.
    """
    def get_rate(self, T, norm):
        return norm * np.array(E_BAR) ** -0.5 * np.exp(-E_BAR / T)


class GaussianAbsorptionLine(SpectralComponent):
    """
    Absorption line centred at mu w std sigma.
    """
    def get_rate(self, power, mu, sigma):
        return -power * _normal_pdf(E_BAR, mu, sigma)


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
        return rate


if __name__ == '__main__':
    c1 = PowerLaw()
    c1args = (1,1)
    c2 = GaussianEmissionLine()
    c2args = (0.1, 5, 0.05)
    self = Spectrum(c1, c2)
    from matplotlib import pyplot as plt
    plt.plot(self.get_rate(c1args, c2args))
    a = np.zeros_like(self.get_rate(c1args, c2args))
    """
    plt.plot(self.spectrum)
    plt.show()
    plt.plot(self.rate)
    plt.show()"""