import numpy as np
import torch
from sbi import utils as utils
from sbi import analysis as analysis
from utils.data_formats import timeseries_to_channels
from simulators import Simulator
from spectralcomponents import PowerLaw, GaussianEmissionLine, Spectrum
from sbi.inference import SNPE, SNRE_A, SNLE, prepare_for_sbi, simulate_for_sbi
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi_tools import BoxUniform
from sbi.utils import posterior_nn, likelihood_nn, classifier_nn
from matplotlib import pyplot as plt
from get_raw_data import simulate_simple
from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


def evaluate_c2st(true_samples, est_samples, seed=None, n_folds=5):
    if torch.is_tensor(true_samples):
        true_samples = true_samples.detach().cpu().numpy()
    if torch.is_tensor(est_samples):
        est_samples = est_samples.detach().cpu().numpy()
    ndim = true_samples.shape[1]
    data = np.concatenate((true_samples, est_samples))
    target = np.concatenate(
        (
            np.zeros((true_samples.shape[0],)),
            np.ones((est_samples.shape[0],)),
        )
    )
    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(10 * ndim, 10 * ndim),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )
    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return torch.from_numpy(np.atleast_1d(scores))


def evaluate_coverage(
        posterior,
        simulate_func,
        total_trials=20, 
        confidence_intervals=np.array([0.05, 0.33, 0.5]), 
        n_samples=100, 
        true_params=None, 
        prior=None):
    """Currently only for 2 params."""
    if true_params is None:
        true_params = prior.sample()
    num_params = len(true_params)
    total_correct = np.zeros((len(confidence_intervals), num_params))
    
    for _ in tqdm(range(total_trials)):
        x_0 = simulate_func(true_params)
        posterior.set_default_x(x_0)
        samples = posterior.sample((n_samples,), x=x_0, show_progress_bars=False)
        
        for i, ci in enumerate(confidence_intervals):
            lower_bounds = np.percentile(
                samples, 
                100 * (ci/2), 
                axis=0)
            upper_bounds = np.percentile(
                samples, 
                100 * (1 - (ci/2)), 
                axis=0)
            
            in_interval = (lower_bounds < true_params) \
            & (true_params < upper_bounds)
            
            total_correct[i] += in_interval

    # Normalize to get frequencies
    correct_freq = total_correct / total_trials
    return correct_freq