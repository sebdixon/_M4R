import numpy as np
import os

base_dir = os.path.dirname(__file__)

arf_path = os.path.join(base_dir, 'arf.txt')
energy_bins_path = os.path.join(base_dir, 'energy_bins.txt')
rmf_path = os.path.join(base_dir, 'rmf.txt')

ARF = np.loadtxt(arf_path)
ENERGY_BINS = np.loadtxt(energy_bins_path)
RMF = np.loadtxt(rmf_path)

def process_rmf(in_RMF):
    in_RMF = (in_RMF.T / in_RMF.sum(axis=1)).T
    return in_RMF

RMF = process_rmf(RMF)
