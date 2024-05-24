import numpy as np
import torch
import os

from torch import nn
from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE, SNPE_C, SNRE_A, simulate_for_sbi, MCMCPosterior
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.utils import posterior_nn, likelihood_nn, classifier_nn
from tqdm import tqdm

from get_raw_data import simulate_simple
from sbi_tools import BoxUniform
from spectralcomponents import PowerLaw, Spectrum
from utils.torch_utils import get_device
from create_posteriors import PosteriorTrainer


# TODO: add docstrings, add GPU support
# TODO: add logging, clarify removal of thinning


# Power law globals
c1 = PowerLaw()
spectrum = Spectrum(c1)

# torch globals
device = get_device()


class SequentialPosteriorTrainer(PosteriorTrainer):
    def __init__(
            self, 
            method: str = 'SNPE', 
            prior = None,
            filepath: str = "simulated_data/power_law",
            embedding_net: str | nn.Module = None,
            model: str = "maf", 
            model_params: tuple | torch.Tensor = None,
            x0: torch.Tensor = None,
            spectrum: Spectrum = spectrum
    ) -> None:
        """
        Sequential posterior trainer class for power law model.
        Saves posteriors to specified filename. 
        Posteriors may then be sampled from.
        This class is meant to be used to train a posterior sequentially.
        ONLY WORKS FOR POWER LAW, SNPE_C and SNRE_A.

        Parameters
        ----------
        method : str
            Inference method to use. Must be one of 'SNPE', 'SNRE'.
            SNPE will train estimator as in Greenberg et al. (2019).
            SNRE will train a classifier as in Hermans et al. (2020).
        prior : torch.distribution
            Prior distribution to use.
        filepath : str
            Filepath to load data from.
        embedding_net : str | nn.Module
            Embedding network to use.
        model : str
            Model to use.
        """
        super().__init__(
            method=method,
            prior=prior,
            filepath=filepath,
            embedding_net=embedding_net,
            model=model,
            model_params=model_params
        )
        self.x0 = x0
        self.spectrum = spectrum
    
    def train(
            self, 
            n_rounds, 
            n_sims,
    ):
        sims_per_round = n_sims // n_rounds
        inference_method = self.get_initial_architecture()
        proposal = self.prior
        posteriors = []
        for i in range(n_rounds):
            print (f'Simulating round {i}')
            if i == 0 and sims_per_round == 1000:
                theta = torch.Tensor(np.load('simulated_data/power_law/theta_chunk1_power_law.npy'))
                x = torch.Tensor(np.load('simulated_data/power_law/x_chunk1_power_law.npy'))
            elif i == 0 and sims_per_round == 2000:
                theta1 = torch.Tensor(np.load('simulated_data/power_law/theta_chunk1_power_law.npy'))
                x1 = torch.Tensor(np.load('simulated_data/power_law/x_chunk1_power_law.npy'))
                theta2 = torch.Tensor(np.load('simulated_data/power_law/theta_chunk2_power_law.npy'))
                x2 = torch.Tensor(np.load('simulated_data/power_law/x_chunk2_power_law.npy'))
                theta = torch.cat((theta1, theta2), dim=0)
                x = torch.cat((x1, x2), dim=0)
                print (x.shape)
                print (theta.shape)
            else:
                theta, x = self.get_data(sims_per_round, proposal)
            if self.method == 'SNPE':
                density_estimator =  inference_method.append_simulations(
                    theta, x, proposal=proposal
                ).train(
                    training_batch_size=100,
                    show_train_summary=True
                )
            elif self.method == 'SNRE':
                density_estimator = inference_method.append_simulations(
                    theta, x, from_round=i
                ).train(
                    training_batch_size=100,
                    retrain_from_scratch=False,
                    discard_prior_samples=False,
                    show_train_summary=True
                )
            print (f'Train round {i}')
            posterior = inference_method.build_posterior(density_estimator)
            proposal = posterior.set_default_x(self.x0)
            self.save_posterior(proposal, i + 1)
            if i >= 1:
                if self.method == 'SNRE':
                    posterior.init_strategy = 'latest_sample'
                    posterior._mcmc_init_params = posteriors[-1]._mcmc_init_params
                self.save_samples(x, i)
                self.save_posterior_samples(theta, i)
            posteriors.append(posterior)
        final_theta, final_x = self.get_data(sims_per_round, proposal)
        self.save_posterior_samples(final_theta, n_rounds)
        self.save_samples(final_x, n_rounds)
        

    def get_data(self, n_sims, proposal):
        if self.method == 'SNRE':
            thetas = proposal.sample(
                (n_sims,),
                thin=1,
                method='slice_np',
                warmup_steps=100,
                x=self.x0
                )
        else:
            thetas = proposal.sample(
                (n_sims,),
                )
        xs = torch.zeros((n_sims, 1024))
        for i in tqdm(range(n_sims), desc='Simulating data'):
            xs[i] = torch.Tensor(simulate_simple(
                spectrum, 
                thetas[i]
            ))
        return thetas, xs

    
    def save_posterior_samples(self, theta, chunk):
        torch.save(
            theta, 
            f'{self.filepath}/posteriorsequential{self.method}_chunk{chunk}_theta_power_law.pt'
        )

    def save_posterior(self, posterior, chunk: int = 0) -> None:
        filepath = f'{self.filepath}/posteriorsequential{self.method}_chunk{chunk}_power_law.pt'
        torch.save(posterior, filepath)

    def save_samples(self, x, chunk):
        torch.save(
            x, 
            f'{self.filepath}/sequential{self.method}_chunk{chunk}_x_power_law.pt'
        )

    def get_initial_architecture(self):
        if self.method == 'SNPE':
            neural_posterior = posterior_nn(
                model=self.model, 
                embedding_net=self.embedding_net, 
                hidden_features=100,
                num_transforms=5
            )
            inference = SNPE(
                prior=self.prior,
                density_estimator=neural_posterior
            )
        elif self.method == 'SNRE':
            classifier = classifier_nn(
                model='resnet',
                embedding_net_x=self.embedding_net,
                hidden_features=100
            )
            inference = SNRE_A(
                prior=self.prior,
                classifier=classifier
            )
        return inference


def create_estimate_sequential_posteriors():
    prior = BoxUniform(
        low=torch.tensor([0.1, 0.1]), 
        high=torch.tensor([2, 2])
    )
    embedding_net = FCEmbedding(
        input_dim=1024, 
        output_dim=10, 
        num_layers=3,
        num_hiddens=200
    )
    for method in ['SNPE',]:
        trainer = SequentialPosteriorTrainer(
            method=method,
            prior=prior,
            embedding_net=embedding_net,
            model="maf",
            filepath='simulated_data/power_law/sequentialv2',
            x0=torch.Tensor(np.load('simulated_data/power_law/x0_power_law.npy'))
        )
        trainer.train(
            n_rounds=10, 
            n_sims=20000,
        )


if __name__ == '__main__':
    create_estimate_sequential_posteriors()
    print('done')