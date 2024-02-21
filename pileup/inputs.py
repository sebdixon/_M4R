import numpy as np

# Load the data
ARF = np.loadtxt('txt_inputs/arf.txt')       # Effective area
ENERGY_BINS = np.loadtxt('txt_inputs/energy_bins.txt')
RMF = np.loadtxt('txt_inputs/rmf.txt')

RMF = (RMF.T / RMF.sum(axis=1)).T

