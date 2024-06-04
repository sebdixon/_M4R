import numpy as np
import torch
import os

from torch import nn
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE, SNRE_A, SNLE_A, DirectPosterior
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.utils import posterior_nn, likelihood_nn, classifier_nn

from get_raw_data import simulate_simple
from sbi_tools import BoxUniform
from spectralcomponents import PowerLaw, Spectrum
from utils.torch_utils import get_device


# Power law globals
c1 = PowerLaw()
spectrum = Spectrum(c1)

# torch globals
device = get_device()


def save_posterior(inference: SNPE, chunk: int) -> None:
    inference_copy = inference
    _ = inference_copy.train(force_first_round_loss=True)
    torch.manual_seed(0)
    posterior = inference_copy.build_posterior()
    torch.save(posterior, f'simulated_data/power_law/posterior_chunk{chunk}_power_law.pt')


class PosteriorTrainer():
    def __init__(
            self, 
            method: str = 'SNPE', 
            prior = None,
            filepath: str = "simulated_data/power_law",
            embedding_net: str | nn.Module = None,
            model: str = "nsf", 
            model_params: tuple | torch.Tensor = None
    ) -> None:
        """
        Posterior trainer class for power law model.
        Saves posteriors to specified filename. 
        Posteriors may then be sampled from.

        Parameters
        ----------
        method : str
            Inference method to use. Must be one of 'SNPE', 'SNLE', 'SNRE'.
        prior : torch.distribution
            Prior distribution to use.
        filepath : str
            Filepath to load data from.
        embedding_net : str | nn.Module
            Embedding network to use.
        model : str
            Model to use.
        model_params : tuple | torch.Tensor
            Model parameters to use.
        """
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
                output_dim=10,
                num_layers=3,
                num_hiddens=100
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
        

    def initialise_SNPE(
            self, 
            model_params: dict
    ) -> None:
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
    
    def initialise_SNLE(
            self, 
            model_params: dict
    ) -> None:
        self.method = 'SNLE'
        neural_likelihood = likelihood_nn(
            model='maf',
            hidden_features=200, 
            num_transforms=5
        )
        self.inference = SNLE_A(
            prior=self.prior,
            density_estimator=neural_likelihood
        )
    
    def initialise_SNRE(
            self, 
            model_params: dict
    ) -> None:
        self.method = 'SNRE'
        classifier = classifier_nn(
            model='resnet',
            embedding_net_x=self.embedding_net,
            hidden_features=200,
        )
        self.inference = SNRE_A(
            prior=self.prior,
            classifier=classifier
        )
    
    def load_data_in_chunks(
            self, 
            chunks: int = 10, 
            filepath: str = None
    ) -> None:
        if filepath is None:
            filepath = self.filepath
        for chunk in range(1, chunks + 1):
            theta = np.load(f'{filepath}/theta_chunk{chunk}_power_law.npy')
            x = np.load(f'{filepath}/x_chunk{chunk}_power_law.npy')
            theta = torch.Tensor(theta)
            x = torch.Tensor(x)
            self.inference.append_simulations(theta, x)

    def train(self) -> DirectPosterior:
        self.load_data_in_chunks()
        self.inference.train()
        if self.method == 'SNLE':
            mcmc_parameters = {"num_chains": 20,
                   "thin": 2}
            self.posterior = self.inference.build_posterior(
                mcmc_method="slice_np_vectorized",
                mcmc_parameters=mcmc_parameters)
        self.posterior = self.inference.build_posterior()
        return self.posterior
        

    def save_posterior(
            self,
            filepath: str = '',
            chunk: int = 0
    ) -> None:
        if filepath == '':
            raise ValueError('filepath must be passed')
        torch.save(
            self.posterior, 
            f'{filepath}posterior{self.method}_{chunk}k_sims.pt')


def create_estimate_posteriors_v2(
        filepath=None,
        methods=None,
        chunks=None
    ) -> None:
    """
    Create estimate posteriors for power law model.
    Uses prespecified model parameters.
    """
    if filepath is None:
        filepath = 'simulated_data/power_law/'
    if methods is None:
        methods = ['SNPE', 'SNLE', 'SNRE']
    if chunks is None:
        chunks = [1, 5, 10]
    for method in methods:
        for chunk in chunks:
            posterior = PosteriorTrainer(
                method=method,
                filepath=filepath,
                model="maf",
                model_params={'hidden_features': 200,
                                'num_transforms': 5}
            )
            posterior.load_data_in_chunks(chunks=chunk)
            _ = posterior.train()
            posterior.save_posterior(
                filepath=filepath, 
                chunk=chunk)


if __name__ == "__main__":
    #test_architecture('SNRE')
    create_estimate_posteriors_v2(
        filepath='simulated_data/power_law/v2',
        methods=['SNRE',],
        chunks=[100]
    )
    # simulate reference x0
    # if file doesnt exist already at location
    if not os.path.exists('simulated_data/power_law/x0_power_law.npy'):
        c1 = PowerLaw()
        spectrum = Spectrum(c1)
        true_params = torch.tensor([0.5, 1.5])
        x0 = simulate_simple(spectrum, true_params)
        np.save('simulated_data/power_law/x0_power_law.npy', x0)
    else:
        x0 = np.load('simulated_data/power_law/x0_power_law.npy')
    # load posteriors
    # for method in ['SNRE']:
    #     for sims in [100]:
    #         posterior = torch.load(f'simulated_data/power_law/posterior{method}_{sims}k_sims.pt')
    #         posterior.set_default_x(x0)
    #         samples = posterior.sample((10000,), x=x0, show_progress_bars=True)
    #         np.save(f'simulated_data/power_law/posterior_samples_{method}_{sims}k_sims.npy', samples)

    for method in ['SNRE',]:
        for sims in [100]:
            posterior = torch.load(f'simulated_data/power_law/v2posterior{method}_{sims}k_sims.pt')
            posterior.set_default_x(x0)
            samples = posterior.sample((1000,), x=x0, show_progress_bars=True)
            np.save(f'simulated_data/power_law/v2posterior_samples_{method}_{sims}k_sims.npy', samples)



    posterior = torch.load('simulated_data/power_law/v2posteriorSNRE_10k_sims.pt')
    samples = posterior.sample(
        (10000,), 
        x=x0, 
        show_progress_bars=True,
        thin=1,
        warmup_steps=1000)
    analysis.pairplot(samples)
    samples = np.load('simulated_data/power_law/posterior_samples_AMHMCMC_1.npy')

    true_samples = np.load('simulated_data/power_law/posterior_samples_AMHMCMC_0.npy')