# code adapted from https://github.com/wjmaddox/drbayes/blob/master/subspace_inference/posteriors/vi_model.py

import math

import torch

from utils import extract_parameters
from utils import set_weights_old as set_weights


class VIModel(torch.nn.Module):

    def __init__(self,
                 base,
                 subspace,
                 init_inv_softplus_sigma=-3.0,
                 prior_log_sigma=3.0,
                 eps=1e-6,
                 with_mu=True,
                 *args,
                 **kwargs):
        super(VIModel, self).__init__()

        self.base_model = base(*args, **kwargs)
        self.base_params = extract_parameters(self.base_model)

        self.subspace = subspace
        self.rank = self.subspace.rank

        self.prior_log_sigma = prior_log_sigma
        self.eps = eps

        self.with_mu = with_mu
        if with_mu:
            self.mu = torch.nn.Parameter(torch.zeros(self.rank))
        self.inv_softplus_sigma = torch.nn.Parameter(
            torch.empty(self.rank).fill_(init_inv_softplus_sigma))

    def forward(self, *args, **kwargs):
        device = self.inv_softplus_sigma.device
        sigma = torch.nn.functional.softplus(
            self.inv_softplus_sigma) + self.eps
        if self.with_mu:
            z = self.mu + torch.randn(self.rank, device=device) * sigma
        else:
            z = torch.randn(self.rank, device=device) * sigma
        w = self.subspace(z)

        set_weights(self.base_params, w, device)
        #set_weights(self.base_model, w, device)

        return self.base_model(*args, **kwargs)

    def sample(self, scale=1.):
        device = self.inv_softplus_sigma.device
        sigma = torch.nn.functional.softplus(
            self.inv_softplus_sigma.detach()) + self.eps
        if self.with_mu:
            z = self.mu + torch.randn(self.rank, device=device) * sigma * scale
        else:
            z = torch.randn(self.rank, device=device) * sigma * scale
        w = self.subspace(z)
        return w

    #def sample_z(self):
    #    sigma = torch.nn.functional.softplus(self.inv_softplus_sigma.detach().cpu()) + self.eps
    #    z = torch.randn(self.rank) * sigma
    #    if self.with_mu:
    #        z += self.mu.detach().cpu()
    #    return z

    def compute_kl(self):
        sigma = torch.nn.functional.softplus(
            self.inv_softplus_sigma) + self.eps

        kl = torch.sum(self.prior_log_sigma - torch.log(sigma) + 0.5 *
                       (sigma**2) / (math.exp(self.prior_log_sigma * 2)))
        if self.with_mu:
            kl += 0.5 * torch.sum(self.mu**2) / math.exp(
                self.prior_log_sigma * 2)
        return kl

    def compute_entropy(self):
        sigma = torch.nn.functional.softplus(
            self.inv_softplus_sigma) + self.eps
        return torch.sum(torch.log(sigma))


class ELBO(object):

    def __init__(self, criterion, num_samples, temperature=1.):
        self.criterion = criterion
        self.num_samples = num_samples
        self.temperature = temperature

    def __call__(self, model, input, target):

        nll, output, _ = self.criterion(model(input), target)
        kl = model.compute_kl() / self.num_samples
        kl *= self.temperature
        loss = nll + kl

        return loss, output, {'nll': nll.item(), 'kl': kl.item()}
