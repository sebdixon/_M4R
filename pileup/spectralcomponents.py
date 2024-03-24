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
    def __init__(self):
        pass

    def get_rate(self):
        raise NotImplementedError()


class PowerLaw(SpectralComponent):
    """
    Basic power law spectrum. Params alpha + beta.
    """
    @property
    def num_params(self):
        return 2

    def get_rate(self, alpha, beta):
        return np.array(alpha * torch.Tensor(E_BAR) ** -beta)


class GaussianEmissionLine(SpectralComponent):
    """
    Emision line centred at mu w std sigma. Added power
    param as well bc unsure if should exist or not. Considering
    fixing sigma=0.1.
    """
    @property
    def num_params(self):
        return 3

    def get_rate(self, power, mu, sigma):
        return power * _normal_pdf(E_BAR, mu, sigma)
    

class DeltaEmissionLine(SpectralComponent):
    """
    Delta function emission line at energy mu.
    """
    @property
    def num_params(self):
        return 2

    def get_rate(self, power, mu):
        # Need to return a line at the nearest neighbour of mu in E_BAR
        idx = np.abs(E_BAR - mu).argmin()
        rate = np.zeros_like(E_BAR)
        rate[idx] = power
        return rate
        


class BrokenPowerLaw(SpectralComponent):
    """
    Broken power law spectrum. Params alpha1, beta1, alpha2, beta2, break energy.
    """
    @property
    def num_params(self):
        return 5

    def get_rate(self, alpha1, beta1, alpha2, beta2, e_break):
        return np.array(alpha1 * torch.Tensor(E_BAR[E_BAR < e_break]) ** -beta1) + np.array(alpha2 * torch.Tensor(E_BAR[E_BAR >= e_break]) ** -beta2)
    

class BlackBody(SpectralComponent):
    """
    Black body spectrum. Params T, normalization.
    """
    @property
    def num_params(self):
        return 2

    def get_rate(self, T, norm):
        return norm * (8 * np.pi * np.array(E_BAR) ** 2) / (np.exp(E_BAR / T) - 1)
    

class Bremsstrahlung(SpectralComponent):
    """
    Bremsstrahlung spectrum. Params T, normalization.
    """
    @property
    def num_params(self):
        return 2

    def get_rate(self, T, norm):
        return norm * np.array(E_BAR) ** -0.5 * np.exp(-E_BAR / T)


class GaussianAbsorptionLine(SpectralComponent):
    """
    Absorption line centred at mu w std sigma.
    """
    def __init__(self):
        self.num_params = 3
        super().__init__()

    def get_rate(self, power, mu, sigma):
        return -power * _normal_pdf(E_BAR, mu, sigma)


class Spectrum(SpectralComponent):
    def __init__(self, *components):
        self.components = components
    
    def get_rate(self, params):
        """
        Compute the rate by summing up the rates from all components.
        
        :param params: A flat tuple containing all parameters for all components.
        """
        if isinstance(params, torch.Tensor):
            params = tuple(params.cpu().numpy())
        rate = np.zeros_like(E_BAR)
        start_idx = 0
        for component in self.components:
            # Assuming each component has a 'num_params' attribute indicating
            # how many parameters it requires.
            end_idx = start_idx + component.num_params
            component_params = params[start_idx:end_idx]
            rate += component.get_rate(*component_params)
            start_idx = end_idx  # Prepare the start index for the next component
            
        self.spectrum = rate.copy()
        rate *= TIME_WIDTH * BIN_WIDTH * ARF
        return rate


if __name__ == '__main__':
    c1 = PowerLaw()
    c1args = (1,1)
    c2 = DeltaEmissionLine()
    c2args = (1, 5.005)
    self = Spectrum(c1, c2)
    from matplotlib import pyplot as plt
    plt.plot(self.get_rate(torch.Tensor(c1args + c2args)))
    a = np.zeros_like(self.get_rate(c1args + c2args))
    """
    plt.plot(self.spectrum)
    plt.show()
    plt.plot(self.rate)
    plt.show()"""