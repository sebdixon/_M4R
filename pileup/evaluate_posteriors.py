import numpy as np
import torch
import pandas as pd

from scipy.stats import gaussian_kde
from sbi import utils as utils
from sbi import analysis as analysis
from spectralcomponents import PowerLaw, Spectrum
from sbi.inference import DirectPosterior
# diagnostics only in update - need to 
# rework other stuff before update
# from sbi.diagnostics import run_sbc, sbc_rank_plot

from get_raw_data import simulate_simple
from tqdm import tqdm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier


def evaluate_c2st(
        true_samples: np.ndarray | torch.Tensor, 
        est_samples: np.ndarray | torch.Tensor, 
        seed: int = 0, 
        n_folds: int = 5,
        scoring: str = "roc_auc"
) -> np.ndarray:
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
        hidden_layer_sizes=(50 * ndim, 50 * ndim),
        max_iter=100000,
        solver="adam",
        random_state=seed,
    )
    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    scores = np.asarray(np.mean(scores)).astype(np.float32)
    return torch.from_numpy(np.atleast_1d(scores))


def hpd_interval(samples, ci):
    """
    Calculate the highest posterior density (HPD) interval for a given confidence level.

    Args:
        samples (np.ndarray): Posterior samples.
        ci (float): Confidence interval level (e.g., 0.95 for 95%).

    Returns:
        tuple: Lower and upper bounds of the HPD interval.
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    interval_idx_inc = int(np.floor(ci * n))
    n_intervals = n - interval_idx_inc

    interval_width = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
    min_idx = np.argmin(interval_width)
    hpd_min = sorted_samples[min_idx]
    hpd_max = sorted_samples[min_idx + interval_idx_inc]

    return hpd_min, hpd_max

def evaluate_coverage(
        posterior: DirectPosterior,
        simulate_func: callable,
        total_trials: int = 20, 
        n_samples: int = 100, 
        true_params: np.ndarray = None, 
        prior = None
) -> np.ndarray:
    """
    Evaluate coverage for HPD intervals from 1% to 99%.

    Args:
        posterior (DirectPosterior): Posterior object with sampling capabilities.
        simulate_func (callable): Function to simulate data given true parameters.
        total_trials (int): Number of trials to perform (ie number of posteriors to generate)
        n_samples (int): Number of samples to draw from the posterior.
        true_params (np.ndarray): True parameters of the model.
        prior: Prior distribution object to sample true parameters if not provided.

    Returns:
        np.ndarray: Coverage frequencies for each HPD interval and parameter.
    """
    if true_params is None:
        true_params = prior.sample()
    num_params = len(true_params)
    
    # Generate confidence intervals from 1% to 99%
    confidence_intervals = np.arange(0.01, 1.00, 0.01)
    total_correct = np.zeros((len(confidence_intervals), num_params))
    
    for _ in tqdm(range(total_trials)):
        x_0 = simulate_func(true_params)
        posterior.set_default_x(x_0)
        samples = posterior.sample((n_samples,), x=x_0, show_progress_bars=False)
        
        for i, ci in enumerate(confidence_intervals):
            lower_bounds = np.zeros(num_params)
            upper_bounds = np.zeros(num_params)
            for j in range(num_params):
                lower_bounds[j], upper_bounds[j] = hpd_interval(samples[:, j], ci)
            
            in_interval = (lower_bounds < true_params) & (true_params < upper_bounds)
            total_correct[i] += in_interval

    correct_freq = total_correct / total_trials
    return correct_freq



def mvs_region(samples, ci):
    """
    Calculate the minimum volume set (MVS) region for a given confidence level using KDE.

    Args:
        samples (np.ndarray): Posterior samples.
        ci (float): Confidence interval level (e.g., 0.95 for 95%).

    Returns:
        np.ndarray: Boolean array indicating which samples are within the MVS region.
    """
    kde = gaussian_kde(samples.T)
    density = kde(samples.T)
    threshold = np.percentile(density, (1 - ci) * 100)
    
    in_region = density >= threshold
    return in_region

def evaluate_coverage_mvs(
        posterior: DirectPosterior,
        simulate_func: callable,
        total_trials: int = 20, 
        n_samples: int = 100, 
        true_params: np.ndarray = None, 
        prior = None
) -> np.ndarray:
    """
    Evaluate coverage for smallest volume confidence regions from 1% to 99%.

    Args:
        posterior (DirectPosterior): Posterior object with sampling capabilities.
        simulate_func (callable): Function to simulate data given true parameters.
        total_trials (int): Number of trials to perform (ie number of posteriors to generate)
        n_samples (int): Number of samples to draw from the posterior.
        true_params (np.ndarray): True parameters of the model.
        prior: Prior distribution object to sample true parameters if not provided.

    Returns:
        np.ndarray: Coverage frequencies for each smallest volume confidence region and parameter.
    """
    if true_params is None:
        true_params = prior.sample()
    num_params = len(true_params)
    
    # Generate confidence intervals from 1% to 99%
    confidence_intervals = np.arange(0.01, 1.00, 0.01)
    total_correct = np.zeros((len(confidence_intervals), num_params))
    
    for _ in tqdm(range(total_trials)):
        x_0 = simulate_func(true_params)
        posterior.set_default_x(x_0)
        samples = posterior.sample((n_samples,), x=x_0, show_progress_bars=False)
        
        for i, ci in enumerate(confidence_intervals):
            in_region = mvs_region(samples, ci)
            
            for j in range(num_params):
                correct_in_region = samples[in_region, j]
                if np.any(correct_in_region == true_params[j]):
                    total_correct[i, j] += 1

    correct_freq = total_correct / total_trials
    return correct_freq


def evaluate_metric(
        test_samples_filepaths: list[str],
        reference_samples_filepath: str,
        metric: str,
        **kwargs
) -> list[np.ndarray]:
    if metric == 'ppc':
        return evaluate_ppc(
            test_samples_filepaths,
            **kwargs
        )
    elif metric == 'c2st':
        metric = evaluate_c2st
    elif metric == 'coverage':
        metric = evaluate_coverage
    test_samples = [
        np.load(filepath) if filepath[-4:] == '.npy'
        else torch.load(filepath).detach().cpu().numpy()
        for filepath in test_samples_filepaths
    ]
    reference_sample = np.load(reference_samples_filepath)
    scores = []
    for test_sample in tqdm(test_samples, desc=f'Evaluating metric {metric}'):
        scores.append(metric(test_sample, reference_sample, **kwargs))
    return scores


def ppc(
        posterior_samples: np.ndarray | torch.Tensor,
        observed_data: np.ndarray | torch.Tensor,
        simulate_func: callable,
        spectrum: Spectrum = None,
        true_params: np.ndarray | torch.Tensor = None
) -> float:
    """Calculate median distance of simulations from posterior
    from the observed data."""
    if true_params is None:
        true_params = torch.Tensor([0.5, 1.5])
    if spectrum is None:
        c1 = PowerLaw()
        spectrum = Spectrum(c1)
    num_params = len(true_params)
    distances = np.zeros(len(posterior_samples))
    for i, params in enumerate(tqdm(posterior_samples, total=len(posterior_samples), desc='PPC')):
        x0 = simulate_func(spectrum, params)
        distances[i] = np.linalg.norm(x0 - observed_data)
    median_distance = np.median(distances)
    return median_distance


def evaluate_ppc(
        test_samples_filepaths: list[str],
        observed_data_filepath: str,
        simulate_func: callable
) -> list[float]:
    test_samples = [
        np.load(filepath) if filepath[-4:] == '.npy'
        else torch.load(filepath).detach().cpu().numpy()
        for filepath in test_samples_filepaths
    ]
    observed_data = np.load(observed_data_filepath)
    ppcs = []
    for test_sample in test_samples:
        ppcs.append(ppc(test_sample, observed_data, simulate_func))
    return ppcs


def create_amortised_dfs(
        test_samples_filepaths: list[str],
        reference_samples_filepath: str,
        metric: str,
        **kwargs
) -> pd.DataFrame:
    scores = evaluate_metric(
        test_samples_filepaths,
        reference_samples_filepath,
        metric,
        **kwargs
    )
    df = pd.DataFrame({
        'Method': ['NLE']*3 + ['NRE']*3 + ['NPE']*3,
        'Samples': [1, 5, 10]*3,
        'Score': np.array([score.item() if score.item() > 0.5 else 1 - score.item() for score in scores])
    })
    return df


def evaluate_amortised_posteriors(
        metrics: list[str] = ['c2st', 'ppc']
):
    # get rid of this / make it more general
    test_samples_filepaths = [
        f'simulated_data/power_law/posterior_samples_SN{method}E_{sims}k_sims.npy'
        for method in ['R', 'P', 'L']
        for sims in [1, 5, 10]
    ]
    reference_samples_filepath = 'simulated_data/power_law/posterior_samples_AMHMCMC_1.npy'
    observed_data_filepath = 'simulated_data/power_law/x0_power_law.npy'
    
    if 'c2st' in metrics:
        metric = 'c2st'
        scores = evaluate_metric(
        test_samples_filepaths, 
        reference_samples_filepath, 
        metric)
        print (scores)
        df = pd.DataFrame({
            'Method': ['NRE']*3 + ['NPE']*3 + ['NLE']*3,
            'Samples': [1, 5, 10]*3,
            'Score': np.array([score.item() if score.item() > 0.5 else 1 - score.item() for score in scores])
        })
        df.to_csv('simulated_data/power_law/c2st_scores.csv', index=False)

    if 'ppc' in metrics:
        dist = evaluate_metric(
            test_samples_filepaths,
            None,
            'ppc',
            simulate_func=simulate_simple,
            observed_data_filepath=observed_data_filepath,
        )
        print (dist)
        df = pd.DataFrame({
            'Method': ['NRE']*3 + ['NPE']*3 + ['NLE']*3,
            'Samples': [1, 5, 10]*3,
            'Score': np.array(dist)
        })
        # save df
        df.to_csv('simulated_data/power_law/meddist_scores.csv', index=False)
        baseline_meddist = ppc(
            np.load('simulated_data/power_law/posterior_samples_AMHMCMC_1.npy'),
            np.load('simulated_data/power_law/x0_power_law.npy'),
            simulate_simple
        )
        print (baseline_meddist)


def ppc_sequential_posteriors(
        test_simulations_filepaths: list[str],
        observed_data_filepath: str
) -> list[float]:
    # i dont really want this to have to be a separate function
    test_samples = [
        torch.load(filepath).detach().cpu().numpy()
        for filepath in test_simulations_filepaths
    ]
    observed_data = np.load(observed_data_filepath)
    ppcs = []
    for test_sample in test_samples:
        dist = np.median(np.linalg.norm(test_sample - observed_data, axis=1))
        ppcs.append(dist)
    return ppcs


def evaluate_sequential_posteriors(
        metrics: list[str] = ['c2st'],
        filepath='simulated_data/power_law/sequential'):
    test_samples_filepaths = [
        f'{filepath}/posteriorsequentialSN{algorithm}E_chunk{i}_theta_power_law.pt'
        for algorithm in ['R', 'P']
        for i in range(1, 11)
    ]
    test_simulations_filepaths = [
        f'{filepath}/sequentialSN{algorithm}E_chunk{i}_x_power_law.pt'
        for algorithm in ['R', 'P']
        for i in range(1, 11)
    ]
    reference_samples_filepath = 'simulated_data/power_law/posterior_samples_AMHMCMC_1.npy'
    observed_data_filepath = 'simulated_data/power_law/x0_power_law.npy'

    if 'c2st' in metrics:
        metric = 'c2st'
        scores = evaluate_metric(
            test_samples_filepaths,
            reference_samples_filepath,
            metric
        )
        print (scores)
        df = pd.DataFrame({
            'Method':  ['SNRE']*10 + ['SNPE']*10,
            'Samples': list(range(1, 11)) * 2,
            'Score': np.array([score.item() if score.item() > 0.5 else 1 - score.item() for score in scores])
        })
        df.to_csv(f'{filepath}/c2st_scores_SNPE.csv', index=False)

    if 'ppc' in metrics:
        dist = ppc_sequential_posteriors(
            test_simulations_filepaths=test_simulations_filepaths,
            observed_data_filepath=observed_data_filepath)
        print (dist)
        df = pd.DataFrame({
            'Method': ['SNRE']*10 + ['SNPE']*10,
            'Samples': list(range(1, 11)) * 2,
            'Score': np.array(dist)
        })
        # save df
        df.to_csv(f'{filepath}/meddist_scores_SNPE.csv', index=False)



def evaluate_amortised_posteriors_v2():
    test_samples_filepaths = [
        f'simulated_data/power_law/v2posterior_samples_SN{method}E_{sims}_sims.npy'
        for method in ['R', 'P']
        for sims in ['1k', '10k', '100k']
    ]
    reference_samples_filepath = 'simulated_data/power_law/posterior_samples_AMHMCMC_1.npy'

    observed_data_filepath = 'simulated_data/power_law/x0_power_law.npy'

    metric = 'c2st'
    scores = evaluate_metric(
        test_samples_filepaths,
        reference_samples_filepath,
        metric
    )
    print (scores)
    df = pd.DataFrame({
        'Method': ['NRE']*3 + ['NPE']*3,
        'Samples': [1, 10, 100]*2 ,
        'Score': np.array([score.item() if score.item() > 0.5 else 1 - score.item() for score in scores])
    })
    df.to_csv('simulated_data/power_law/v2c2st_scores.csv', index=False)
    dist = evaluate_metric(
        test_samples_filepaths,
        None,
        'ppc',
        simulate_func=simulate_simple,
        observed_data_filepath=observed_data_filepath,
    )
    print (dist)
    df = pd.DataFrame({
        'Method': ['NRE']*3 + ['NPE']*3,
        'Samples': [1, 10, 100]*2 ,
        'Score': np.array(dist)
    })
    # save df
    df.to_csv('simulated_data/power_law/v2meddist_scores.csv', index=False)




if __name__ == '__main__':

    evaluate_amortised_posteriors(
        metrics=['c2st'])
    evaluate_sequential_posteriors()
    # evaluate_sequential_posteriors(
    #     filepath='simulated_data/power_law/sequentialv2'
    # )
    evaluate_amortised_posteriors_v2()

    posterior = torch.load('simulated_data/power_law/v2posteriorSNPE_100k_sims.pt')
    c1 = PowerLaw()
    spec = Spectrum(c1)
    simulate_func = lambda params: simulate_simple(spec, params)
    coverage_vals = evaluate_coverage(
        posterior,
        simulate_func,
        total_trials=1000,
        n_samples=10000,
        true_params=np.array([0.5, 1.5])
    )
    np.save('simulated_data/power_law/coverage_vals.npy', coverage_vals)
    coverage_vals_mvs = evaluate_coverage_mvs(
        posterior,
        simulate_func,
        total_trials=1000,
        n_samples=10000,
        true_params=np.array([0.5, 1.5])
    )

    samples = np.load('simulated_data/power_law/posterior_samples_SNPE_10k_sims.npy')
    analysis.pairplot(samples, labels=['alpha', 'beta'])
    true_samples = np.load('simulated_data/power_law/posterior_samples_AMHMCMC_0.npy')
    analysis.pairplot(true_samples, labels=['alpha', 'beta'])

    import plotly.express as px
    import pandas as pd
    # plot df, 1 plot for each method
    # i want subplots for each of NLE, NPE, NRE

    # posterior = torch.load('simulated_data/power_law/posteriorSNPE_1k_sims.pt')
    # data = np.load('simulated_data/power_law/x0_power_law.npy')
    # ps = posterior.set_default_x(data).sample((1000,), x=data, show_progress_bars=True)

    # evaluate_c2st(ps, np.load('simulated_data/power_law/posterior_samples_AMHMCMC.npy'))

    p = torch.load('simulated_data/power_law/sequential/v2posteriorsSNPE_chunk5_power_law.pt')
    ps = np.load('simulated_data/power_law/v2posterior_samples_SNRE_5k_sims.npy')
    x0 = np.load('simulated_data/power_law/x0_power_law.npy')
    ps = p.set_default_x(x0).sample((1000,), x=x0, show_progress_bars=True)
    analysis.pairplot(ps, labels=['alpha', 'beta'])