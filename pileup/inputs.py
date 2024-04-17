import numpy as np

# Load the data
ARF = np.loadtxt('txt_inputs/arf.txt')       # Effective area
ENERGY_BINS = np.loadtxt('txt_inputs/energy_bins.txt')
RMF = np.loadtxt('txt_inputs/rmf.txt')

def process_rmf(in_RMF):
    in_RMF = (in_RMF.T / in_RMF.sum(axis=1)).T
    return in_RMF

RMF = process_rmf(RMF)