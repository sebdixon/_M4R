import torch
import numpy as np

from tqdm import tqdm

class SequentialRatioEstimator:


    def __init__(
            self,
            simulator:object,
            prior:torch.distributions.Distribution,
            true_obs:torch.Tensor,
            classifier,
            n_rounds:int=20
    ):
        self.simulator = simulator
        self.prior = prior
        self.true_obs = true_obs
        self.classifier = classifier
        self.n_rounds = n_rounds
        self._params = []
        self._obs = []


    def infer(self, n_samples):
        for round in tqdm(range(self.n_rounds)):
            if round == 0:
                params = self.prior.sample((n_samples,))
            else:
                params = self._sample_posterior((n_samples,))
            obs = self.sample_observations(params)
            self._params.append(params)
            self._obs.append(obs)
    

    def _sample_posterior(self):
        raise NotImplemented
    
    def _sample_observations(self):
        raise NotImplemented


