import numpy as np
import torch
from sbi import utils as utils
from sbi import analysis as analysis
from utils.data_formats import timeseries_to_channels
from simulators import Simulator
from spectralcomponents import PowerLaw, GaussianEmissionLine, Spectrum
from sbi.inference import SNPE, SNRE_A, SNLE_A, SNLE, prepare_for_sbi, simulate_for_sbi
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi_tools import BoxUniform
from sbi.utils import posterior_nn, likelihood_nn, classifier_nn
from matplotlib import pyplot as plt
from get_raw_data import simulate_simple


c1 = PowerLaw()
spectrum = Spectrum(c1)

def save_posterior(inference: SNPE, chunk):
    inference_copy = inference
    _ = inference_copy.train(force_first_round_loss=True)
    torch.manual_seed(0)
    posterior = inference_copy.build_posterior()
    torch.save(posterior, f'simulated_data/power_law/posterior_chunk{chunk}_power_law.pt')


class PosteriorTrainer():
    def __init__(
            self, 
            method='SNPE', 
            prior=None,
            filepath="simulated_data/power_law",
            embedding_net=None,
            model="nsf", 
            model_params=None
            ):
        if prior is None:
            self.prior = BoxUniform(
                low=torch.tensor([0.1, 0.1]), 
                high=torch.tensor([2, 2]))
        else:
            self.prior = prior
        self.filepath = filepath
        if embedding_net is None:
            self.embedding_net = FCEmbedding(
                input_dim=1024,
                output_dim=100,
                num_layers=3,
                num_hiddens=1024
            )
        else:
            self.embedding_net = embedding_net
        self.model = model
        if model_params is None:
            model_params = {'hidden_features': 200,
                            'num_transforms': 5}
        if method == 'SNPE' or method is None:
            self.initialise_SNPE(model_params)
        elif method == 'SNLE':
            self.initialise_SNLE(model_params)
        elif method == 'SNRE':
            self.initialise_SNRE(model_params)
        else:
            raise ValueError('method must be one of "SNPE", "SNLE", "SNRE"')
        

    def initialise_SNPE(self, model_params):
        self.method = 'SNPE'
        neural_posterior = posterior_nn(
            model=self.model,
            embedding_net=self.embedding_net,
            **model_params
        )
        self.inference = SNPE(
            prior=self.prior,
            density_estimator=neural_posterior
        )
    
    def initialise_SNLE(self):
        self.method = 'SNLE'
        neural_likelihood = likelihood_nn(
            model='maf',#self.model,
            #embedding_net=self.embedding_net,
            hidden_features=200, 
            num_transforms=5
        )
        self.inference = SNLE_A(
            prior=self.prior,
            density_estimator=neural_likelihood
        )
    
    def initialise_SNRE(self):
        self.method = 'SNRE'
        classifier = classifier_nn(
            model='resnet',
            #embedding_net_x=self.embedding_net,
        )
        self.inference = SNRE_A(
            prior=self.prior,
            classifier=classifier
        )
    
    def load_data_in_chunks(self, chunks=10, filepath=None):
        if filepath is None:
            filepath = self.filepath
        for chunk in range(1, chunks + 1):
            theta = np.load(f'{filepath}/theta_chunk{chunk}_power_law.npy')
            x = np.load(f'{filepath}/x_chunk{chunk}_power_law.npy')
            theta = torch.Tensor(theta)
            x = torch.Tensor(x)
            self.inference.append_simulations(theta, x)

    def train(self):
        self.load_data_in_chunks()
        self.inference.train()
        if self.method == 'SNLE':
            mcmc_parameters = {"num_chains": 20,
                   "thin": 5}
            self.posterior = self.inference.build_posterior(
                mcmc_method="slice_np_vectorized",
                mcmc_parameters=mcmc_parameters)
        self.posterior = self.inference.build_posterior()
        

    def save_posterior(self, filepath='', chunk=0):
        if filepath == '':
            raise ValueError('filepath must be passed')
        torch.save(self.posterior, f'{filepath}/posterior{self.method}_{chunk}k_sims.pt')


def create_reference_posteriors():
    """Creates posteriors on 1k, 5k and 10k sims
    for each of NPE, NRE, NLE
    """
    for method in ['SNLE']:
        for chunk in [1, 5, 10]:
            posterior = PosteriorTrainer(
                method=method,
                filepath='simulated_data/power_law/',
                model="nsf",
                model_params={'hidden_features': 200,
                                'num_transforms': 5}
            )
            posterior.load_data_in_chunks(chunks=chunk)
            posterior.train()
            posterior.save_posterior(filepath='simulated_data/power_law/', chunk=chunk)


if __name__ == "__main__":
    #create_reference_posteriors()
    # simulate reference x0
    c1 = PowerLaw()
    spectrum = Spectrum(c1)
    true_params = torch.tensor([0.5, 1.5])
    x0 = simulate_simple(spectrum, true_params)
    np.save('simulated_data/power_law/x0_power_law.npy', x0)
    # load posteriors

    for method in ['SNPE', 'SNLE', 'SNRE']:
        for sims in [1, 5, 10]:
            posterior = torch.load(f'simulated_data/power_law/posterior{method}_{sims}k_sims.pt')
            posterior.set_default_x(x0)
            samples = posterior.sample((1000,), x=x0, show_progress_bars=True)
            np.save(f'simulated_data/power_law/posterior_samples_{method}_{sims}k_sims.npy', samples)
