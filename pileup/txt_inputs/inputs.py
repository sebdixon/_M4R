import numpy as np

ARF = np.loadtxt('arf.txt') # Effective area
ENERGY_BINS = np.loadtxt('energy_bins.txt')
RMF = np.loadtxt('rmf.txt')
RMF = (RMF.T / RMF.sum(axis=1)).T # Need to normalise RMF
