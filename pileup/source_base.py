# conda deactivate
# conda activate base
import os
import torch
import numpy as np

from sbi import utils as utils
from sbi import analysis as analysis
from sbi.neural_nets.embedding_nets import FCEmbedding
from sbi.inference import SNPE
from sbi.utils import posterior_nn

from sbi_tools import BoxUniform

# globals
ROUND = int(os.getenv('ROUND'))
NUM_SIMS = int(os.getenv('NUM_SIMS'))
X0 = torch.Tensor(np.loadtxt('source_analysis/counts.txt'))


def get_initial_architecture():
    prior = BoxUniform(
            low=torch.tensor([0.1, 0.1]), 
            high=torch.tensor([50, 50])
    )

    embedding_net = FCEmbedding(
        input_dim=1024, 
        output_dim=100, 
        num_layers=3,
        num_hiddens=1024
    )

    neural_posterior = posterior_nn(
        model="nsf", 
        embedding_net=embedding_net, 
        hidden_features=200,
        num_transforms=5
    )
    inference = SNPE(
        prior=prior,
        density_estimator=neural_posterior
    )
    return prior, inference


def train_ciao(round):
    theta = torch.Tensor(
        np.load(f'xspec_outputs/ciao_theta_{round}.npy')
    )
    if round == 1:
        prior, inference = get_initial_architecture()
        proposal = prior
    else:
        inference = torch.load(f'xspec_outputs/ciao_inference_{round-1}.pt')
        proposal = torch.load(f'xspec_outputs/ciao_posterior_{round-1}.pt')

    inference.append_simulations(theta, X0, proposal=proposal)
    _ = inference.train()
    posterior = inference.build_posterior()
    posterior = posterior.set_default_x(X0)

    torch.save(inference, f'xspec_outputs/ciao_inference_{round}.pt')
    torch.save(posterior, f'xspec_outputs/ciao_posterior_{round}.pt')

    proposals = posterior.sample((NUM_SIMS,), x=X0)
    np.save(f'xspec_outputs/ciao__proposals_{round}.npy', proposals)


if __name__ == '__main__':
    print ('base running round ', ROUND)
    train_ciao(ROUND)
    print('done')