import math
import time

import numpy as np
import tabulate
import torch
from posteriors.mf_gaussian_vi import ELBO, VIModel
from posteriors.proj_model import SubspaceModel
from torch import nn
from trainers.base_trainer import BaseTrainer


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.mean(np.log(ps + 1e-12))
    return nll


class VITrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.init_sd = 1.0
        self.prior_sd = 1.0
        self.with_mu = True
        self.temperature = 1.0
        self.num_samples = 30
        self.var_clamp = 1e-6

    def get_subspace_mean_covar(self):
        # TODO figure out what to do here, maybe just start with the end points and the mid point?
        # Could always sample more
        curve_parameters = list(self.model.parameters())
        w = []

        w.append(
            np.concatenate([
                p.data.cpu().numpy().ravel() for p in [
                    curve_parameters[0], curve_parameters[1],
                    curve_parameters[4], curve_parameters[5]
                ]
            ]))
        w.append(
            np.concatenate([
                p.data.cpu().numpy().ravel() for p in [
                    curve_parameters[2], curve_parameters[1],
                    curve_parameters[6], curve_parameters[5]
                ]
            ]))
        w.append(
            np.concatenate([
                p.data.cpu().numpy().ravel() for p in [
                    curve_parameters[3], curve_parameters[1],
                    curve_parameters[7], curve_parameters[5]
                ]
            ]))

        u = w[2] - w[0]
        dx = np.linalg.norm(u)
        u /= dx

        v = w[1] - w[0]
        v -= np.dot(u, v) * u
        dy = np.linalg.norm(v)
        v /= dy

        # u /= math.sqrt(3.0)
        # v /= 1.5

        cov_factor = np.vstack((u[None, :], v[None, :]))

        mean = np.mean(w, axis=0)
        sq_mean = np.mean(np.square(w), axis=0)

        return torch.FloatTensor(mean), torch.clamp(
            torch.FloatTensor(sq_mean) - torch.FloatTensor(mean**2),
            self.var_clamp), torch.FloatTensor(cov_factor)

    def fit_and_eval_posterior(self):
        mean, var, cov_factor = self.get_subspace_mean_covar()

        vi_model = VIModel(
            subspace=SubspaceModel(mean.to(self.device),
                                   cov_factor.to(self.device)),
            init_inv_softplus_sigma=math.log(math.exp(self.init_sd) - 1.0),
            prior_log_sigma=math.log(self.prior_sd),
            with_mu=self.with_mu)

        vi_model = vi_model.to(self.device)
        # print(
        #     utils.eval(loaders["train"],
        #                vi_model,
        #                criterion=losses.cross_entropy))

        elbo = ELBO(self.criterion, len(self.train_loader.dataset),
                    self.temperature)

        self.fit_posterior(vi_model, elbo)
        self.eval_posterior(vi_model)

    def fit_posterior(self, vi_model, elbo):
        #optimizer = torch.optim.Adam([param for param in vi_model.parameters()], lr=0.01)
        self.optimizer = torch.optim.SGD(
            [param for param in vi_model.parameters()],
            lr=self.learning_rate,
            momentum=0.9)
        columns = ['ep', 'acc', 'loss', 'kl', 'nll', 'sigma_1', 'time']

        epoch = 0
        self.base_model = self.model
        self.model = vi_model
        self.criterion = elbo
        for epoch in range(self.epochs):
            time_ep = time.time()
            train_res = self.train_epoch(self.train_loader)
            time_ep = time.time() - time_ep
            sigma_1 = torch.nn.functional.softplus(
                vi_model.inv_softplus_sigma.detach().cpu())[0].item()
            values = [
                '%d/%d' % (epoch + 1, self.epochs), train_res['accuracy'],
                train_res['loss'], train_res['stats']['kl'],
                train_res['stats']['nll'], sigma_1, time_ep
            ]
            if epoch == 0:
                print(
                    tabulate.tabulate([values],
                                      columns,
                                      tablefmt='simple',
                                      floatfmt='8.4f'))
            else:
                print(
                    tabulate.tabulate([values],
                                      columns,
                                      tablefmt='plain',
                                      floatfmt='8.4f').split('\n')[1])

        print(
            "sigma:",
            torch.nn.functional.softplus(
                vi_model.inv_softplus_sigma.detach().cpu()))
        if self.with_mu:
            print("mu:", vi_model.mu.detach().cpu().data)

        # utils.save_checkpoint(args.dir,
        #                       epoch,
        #                       name='vi',
        #                       state_dict=vi_model.state_dict())
        # self.save_model(f'{self.name}_{iter}')

    def eval_posterior(self, vi_model):
        eval_model = self.model

        ens_predictions = np.zeros(len(self.valid_loader.dataset),
                                   self.out_dim)
        targets = np.zeros(len(self.valid_loader.dataset))

        columns = ['iter ens', 'acc', 'nll']

        for i in range(self.num_samples):
            with torch.no_grad():
                w = vi_model.sample()
                offset = 0
                for param in eval_model.parameters():
                    param.data.copy_(w[offset:offset + param.numel()].view(
                        param.size()).to(self.device))
                    offset += param.numel()

            # Don't think we need batchnorm
            # utils.bn_update(loaders['train'],
            #                 eval_model,
            #                 subset=args.bn_subset)

            with torch.no_grad():
                for j, (x, y) in enumerate(self.valid_loader):
                    reshaped_x = x.reshape(x.size(0), 784)
                    y_hat = self.model(reshaped_x.to(self.device))
                    ens_predictions += y_hat
                    targets = y

            values = [
                '%3d/%3d' % (i + 1, self.num_samples),
                np.mean(np.argmax(ens_predictions, axis=1) == targets),
                nll(ens_predictions / (i + 1), targets)
            ]
            table = tabulate.tabulate([values],
                                      columns,
                                      tablefmt='simple',
                                      floatfmt='8.4f')
            if i == 0:
                print(table)
            else:
                print(table.split('\n')[2])

        ens_predictions /= self.num_samples
        ens_acc = np.mean(np.argmax(ens_predictions, axis=1) == targets)
        ens_nll = nll(ens_predictions, targets)

        return ens_acc, ens_nll


class ESSTrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def fit_and_eval_posterior(self):
        raise NotImplementedError()
