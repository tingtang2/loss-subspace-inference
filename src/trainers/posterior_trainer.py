import itertools
import math
import time
from collections import defaultdict

import numpy as np
import tabulate
import torch
import tqdm
from tqdm import trange
from scipy.special import softmax


from models.mlp import NN
from posteriors.mf_gaussian_vi import ELBO, VIModel
from posteriors.proj_model import SubspaceModel
from posteriors.elliptical_slice import elliptical_slice
from trainers.base_trainer import BaseTrainer
import logging


def train_epoch(loader,
                model,
                criterion,
                optimizer,
                cuda=True,
                regression=False,
                verbose=False,
                subset=None):
    loss_sum = 0.0
    stats_sum = defaultdict(float)
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()

    if subset is not None:
        num_batches = int(num_batches * subset)
        loader = itertools.islice(loader, num_batches)

    if verbose:
        loader = tqdm.tqdm(loader, total=num_batches)

    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        input = input.reshape(input.size(0), 784)
        loss, output, stats = criterion(model, input, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.data.item() * input.size(0)
        for key, value in stats.items():
            stats_sum[key] += value * input.size(0)

        if not regression:
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        num_objects_current += input.size(0)

        if verbose and 10 * (i + 1) / num_batches >= verb_stage + 1:
            print('Stage %d/10. Loss: %12.4f. Acc: %6.2f' %
                  (verb_stage + 1, loss_sum / num_objects_current,
                   correct / num_objects_current * 100.0))
            verb_stage += 1

    return {
        'loss': loss_sum / num_objects_current,
        'accuracy':
        None if regression else correct / num_objects_current * 100.0,
        'stats': {
            key: value / num_objects_current
            for key, value in stats_sum.items()
        }
    }


def flatten(lst):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = softmax(outputs[idx])
    ps = np.clip(ps, 1e-12, 1.0)  # Ensure safe log
    nll = -np.mean(np.log(ps))
    return nll


class VITrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.init_sd = 1.0
        self.prior_sd = 1.0
        self.with_mu = False
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

        cov_factor = np.vstack((u[None, :], v[None, :]))

        mean = np.mean(w, axis=0)
        sq_mean = np.mean(np.square(w), axis=0)

        return torch.FloatTensor(mean), torch.clamp(
            torch.FloatTensor(sq_mean) - torch.FloatTensor(mean**2),
            self.var_clamp), torch.FloatTensor(cov_factor)

    def fit_and_eval_posterior(self):
        mean, var, cov_factor = self.get_subspace_mean_covar()

        vi_model = VIModel(
            base=NN(input_dim=self.data_dim,
                    hidden_dim=self.hidden_size,
                    out_dim=self.out_dim,
                    dropout_prob=self.dropout_prob).to(self.device),
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
        optimizer = torch.optim.SGD([param for param in vi_model.parameters()],
                                    lr=0.1,
                                    momentum=0.9)
        columns = ['ep', 'acc', 'loss', 'kl', 'nll', 'sigma_1', 'time']

        epoch = 0
        for epoch in trange(self.epochs):
            time_ep = time.time()
            train_res = train_epoch(self.train_loader, vi_model, elbo,
                                    optimizer)
            time_ep = time.time() - time_ep
            sigma_1 = torch.nn.functional.softplus(
                vi_model.inv_softplus_sigma.detach().cpu())[0].item()
            values = [
                '%d/%d' % (epoch + 1, self.epochs), train_res['accuracy'],
                train_res['loss'], train_res['stats']['kl'],
                train_res['stats']['nll'], sigma_1, time_ep
            ]
            if epoch == 0:
                logging.info(
                    tabulate.tabulate([values],
                                      columns,
                                      tablefmt='simple',
                                      floatfmt='8.4f'))
            else:
                logging.info(
                    tabulate.tabulate([values],
                                      columns,
                                      tablefmt='plain',
                                      floatfmt='8.4f').split('\n')[1])

        logging.info(
            f'sigma: {torch.nn.functional.softplus(vi_model.inv_softplus_sigma.detach().cpu())}')
        if self.with_mu:
            logging.info(f'mu: {vi_model.mu.detach().cpu().data}')

        # utils.save_checkpoint(args.dir,
        #                       epoch,
        #                       name='vi',
        #                       state_dict=vi_model.state_dict())
        
        # self.save_model(f'{self.name}_')

    def eval_posterior(self, vi_model):
        eval_model = NN(input_dim=self.data_dim,
                    hidden_dim=self.hidden_size,
                    out_dim=self.out_dim,
                    dropout_prob=self.dropout_prob).to(self.device)

        ens_predictions = np.zeros((len(self.valid_loader.dataset), self.out_dim))
        targets = np.zeros(len(self.valid_loader.dataset))

        columns = ['iter ens', 'acc', 'nll']

        for i in range(self.num_samples):
            with torch.no_grad():
                w = vi_model.sample()
                offset = 0
                for parameter in eval_model.parameters():
                    size = np.prod(parameter.size())
                    value = w[offset:offset + size].reshape(parameter.size())
                    parameter.data.copy_(value)
                    offset += size


            with torch.no_grad():
                for j, (x, y) in enumerate(self.valid_loader):
                    reshaped_x = x.reshape(x.size(0), 784)
                    y_hat: torch.tensor = eval_model(reshaped_x.to(self.device))
                    ens_predictions = ens_predictions + y_hat.detach().cpu().numpy()
                    targets = y.detach().cpu().numpy()

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
                logging.info(table)
            else:
                logging.info(table.split('\n')[2])

        ens_predictions /= self.num_samples
        ens_acc = np.mean(np.argmax(ens_predictions, axis=1) == targets)
        ens_nll = nll(ens_predictions, targets)
        print("Ensemble NLL:", ens_nll)
        print("Ensemble Accuracy:", ens_acc)

        return ens_acc, ens_nll

def log_pdf(theta, subspace, model, loader, criterion, temperature, device):
    w = subspace(torch.FloatTensor(theta).to(device))
    offset = 0
    for parameter in model.parameters():
        size = np.prod(parameter.size())
        value = w[offset:offset + size].reshape(parameter.size())
        parameter.data.copy_(value)
        offset += size
    model.train()
    with torch.no_grad():
        loss = 0
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)
            reshaped_x = data.reshape(data.size(0), 784)
            batch_loss = criterion(model(reshaped_x), target)
            loss += batch_loss * data.size()[0]
    return -loss.item() / temperature


class ESSTrainer(BaseTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_samples = 30
        self.rank = 2
        self.prior_sd = 1.0
        self.var_clamp = 1e-6
        self.temperature = 1.0
    
    def oracle(self, theta, model, subspace):
        return log_pdf(
            theta,
            subspace=subspace,
            model=model,
            loader=self.train_loader,
            criterion=self.criterion,
            temperature=self.temperature,
            device=self.device
        )
    
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


        mean = np.mean(w, axis=0)
        sq_mean = np.mean(np.square(w), axis=0)
        
        cov_factor = np.vstack((u[None, :], v[None, :]))
        coords = np.dot(cov_factor / np.sum(np.square(cov_factor), axis=1, keepdims=True), (w - mean[None, :]).T).T
        theta = torch.FloatTensor(coords[2, :])

        return torch.FloatTensor(mean), torch.clamp(
            torch.FloatTensor(sq_mean) - torch.FloatTensor(mean**2),
            self.var_clamp), torch.FloatTensor(cov_factor), theta

    def fit_and_eval_posterior(self):
        mean, var, cov_factor, theta = self.get_subspace_mean_covar()

        subspace = SubspaceModel(mean.to(self.device),
                                   cov_factor.to(self.device)).to(self.device)
        self.eval_posterior(subspace, theta)
    
    def eval_posterior(self, subspace, theta):
        eval_model = NN(input_dim=self.data_dim,
                    hidden_dim=self.hidden_size,
                    out_dim=self.out_dim,
                    dropout_prob=self.dropout_prob).to(self.device)

        ens_predictions = np.zeros((len(self.valid_loader.dataset), self.out_dim))
        targets = np.zeros(len(self.valid_loader.dataset))
        columns = ['iter', 'log_prob', 'acc', 'nll', 'time']

        samples = np.zeros((self.num_samples, self.rank))
        rng = np.random.default_rng(self.seed)

        for i in range(self.num_samples):
            time_sample = time.time()
            prior_sample = rng.normal(loc=0.0, scale=self.prior_sd, size=self.rank)
            theta, log_prob = elliptical_slice(initial_theta=theta.cpu().numpy().copy(), prior=prior_sample,
                                                            lnpdf=self.oracle, model=eval_model, subspace=subspace)
            samples[i, :] = theta
            theta = torch.FloatTensor(theta).to(self.device)
            print(theta)
            w = subspace(theta)
            
            offset = 0
            for parameter in eval_model.parameters():
                size = np.prod(parameter.size())
                value = w[offset:offset + size].reshape(parameter.size())
                parameter.data.copy_(value)
                offset += size


            with torch.no_grad():
                for j, (x, y) in enumerate(self.valid_loader):
                    reshaped_x = x.reshape(x.size(0), 784)
                    y_hat: torch.tensor = eval_model(reshaped_x.to(self.device))
                    ens_predictions = ens_predictions + y_hat.detach().cpu().numpy()
                    targets = y.detach().cpu().numpy()

            time_sample = time.time() - time_sample
            values = ['%3d/%3d' % (i + 1, self.num_samples),
                    log_prob,
                    np.mean(np.argmax(ens_predictions, axis=1) == targets),
                    nll(ens_predictions / (i + 1), targets),
                    time_sample]
            table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
            if i == 0:
                print(table)
            else:
                print(table.split('\n')[2])

        ens_predictions /= self.num_samples
        ens_acc = np.mean(np.argmax(ens_predictions, axis=1) == targets)
        ens_nll = nll(ens_predictions, targets)
        print("Ensemble NLL:", ens_nll)
        print("Ensemble Accuracy:", ens_acc)
        
