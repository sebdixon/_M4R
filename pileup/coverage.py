from sbi.analysis import run_sbc, sbc_rank_plot
import numpy as np
import torch
from sbi.inference import SNPE
from matplotlib import pyplot as plt
from scipy.stats import binom

thetas = np.zeros((5000, 2))
xs = np.zeros((5000, 1024))

for batch in range(1, 6):
    theta = np.load(f'simulated_data/power_law/theta_chunk{batch}_power_law.npy')
    x = np.load(f'simulated_data/power_law/x_chunk{batch}_power_law.npy')
    thetas[(batch-1)*1000:batch*1000] = theta
    xs[(batch-1)*1000:batch*1000] = x

thetas = torch.tensor(thetas).float()
xs = torch.tensor(xs).float()

# posterior = torch.load('simulated_data/power_law/v2posteriorSNPE_10k_sims.pt')

# ranks, dap_samples = run_sbc(
#     thetas, xs, posterior, num_posterior_samples=1000, reduce_fns=posterior.log_prob
# )

# np.save('simulated_data/power_law/ranks_power_law.npy', ranks)
# np.save('simulated_data/power_law/dap_samples_power_law.npy', dap_samples)


# posterior = torch.load('simulated_data/power_law/v2posteriorSNPE_5k_sims.pt')

# ranks, dap_samples = run_sbc(
#     thetas, xs, posterior, num_posterior_samples=1000, reduce_fns=posterior.log_prob
# )

# np.save('simulated_data/power_law/ranks_power_law2.npy', ranks)
# np.save('simulated_data/power_law/dap_samples_power_law2.npy', dap_samples)


# posterior = torch.load('simulated_data/power_law/v2posteriorSNPE_1k_sims.pt')

# ranks, dap_samples = run_sbc(
#     thetas, xs, posterior, num_posterior_samples=1000, reduce_fns=posterior.log_prob
# )

# np.save('simulated_data/power_law/ranks_power_law3.npy', ranks)
# np.save('simulated_data/power_law/dap_samples_power_law3.npy', dap_samples)


ranks = np.load('simulated_data/power_law/ranks_power_law3.npy')
dap_samples = np.load('simulated_data/power_law/dap_samples_power_law3.npy')


fig, ax = sbc_rank_plot(
    ranks=ranks,
    num_posterior_samples=1000,
    plot_type="cdf",
    #num_bins=30,
)

fig.savefig('simulated_data/power_law/sbc_rank_plot_power_law.png')

plt.rcParams.update({
    "font.family": "serif"
})

def calculate_coverage(ranks, num_samples):
    expected_coverage = np.arange(1, num_samples + 1) / (num_samples + 1)
    observed_coverage = np.cumsum(np.histogram(ranks, bins=np.arange(num_samples + 1))[0]) / len(ranks)
    return expected_coverage, observed_coverage

def calculate_confidence_intervals(expected_coverage, num_samples, alpha=0.05):
    lower_bound = binom.ppf(alpha / 2, num_samples, expected_coverage) / num_samples
    upper_bound = binom.ppf(1 - alpha / 2, num_samples, expected_coverage) / num_samples
    return lower_bound, upper_bound

def plot_coverage(expected_coverage, observed_coverage, lower_bound, upper_bound):
    plt.figure(figsize=(8, 6))
    plt.plot(expected_coverage, observed_coverage, label='Observed Coverage')
    plt.plot(expected_coverage, expected_coverage, '--', label='$y=x$')
    plt.fill_between(expected_coverage, lower_bound, upper_bound, color='grey', alpha=0.5, label='95\% Confidence Interval')
    plt.xlabel('Expected Coverage')
    plt.ylabel('Observed Coverage')
    plt.title('Expected vs Observed Coverage')
    plt.legend()
    plt.grid(True)
    plt.savefig('simulated_data/power_law/coverage_power_law.png')
    plt.show()

num_posterior_samples = 1000
expected_coverage, observed_coverage = calculate_coverage(ranks, num_posterior_samples)
lower_bound, upper_bound = calculate_confidence_intervals(expected_coverage, len(ranks))
plot_coverage(expected_coverage, observed_coverage, lower_bound, upper_bound)

posterior = torch.load('simulated_data/power_law/v2posteriorSNPE_5k_sims.pt')
x0 = np.load('simulated_data/power_law/x0_power_law.npy')
samples = posterior.sample((10000,), x=x0)
from sbi import analysis
analysis.pairplot(
    samples
)





from analytical import TruePosterior
from spectralcomponents import PowerLaw, Spectrum
from sbi_tools import BoxUniform

c1 = PowerLaw()
spectrum = Spectrum(c1)

thetap = samples[:100]

prior = BoxUniform(
    low=torch.tensor([0.1, 0.1]), 
    high=torch.tensor([2, 2])
)
x0 = np.load('simulated_data/power_law/x0_power_law.npy')
true_posterior = TruePosterior(
    prior=BoxUniform(
        low=torch.tensor([0.1, 0.1]), 
        high=torch.tensor([2, 2])
    ),
    spectrum=spectrum,
    obs=x0,
    pileup='channels'
)

true_log_prob, est_log_prob = np.zeros(len(thetap)),  np.zeros(len(thetap))
from tqdm import tqdm
for i in tqdm(range(len(thetap))[:3]):
    true_log_prob[i] = true_posterior.compute_log_likelihood(thetap[i])
    est_log_prob[i] = posterior.log_prob(thetap[i], x=x0).numpy()

plt.plot(true_log_prob, est_log_prob, 'o')

true_samples = np.load('simulated_data/power_law/posterior_samples_AMHMCMC_0.npy')