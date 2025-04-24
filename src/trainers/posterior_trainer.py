import math
import time
import tabulate

import numpy as np
import torch
from base_trainer import BaseTrainer

from posteriors.mf_gaussian_vi import ELBO, VIModel
from posteriors.proj_model import SubspaceModel

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
        
    def get_subspace_mean_covar(self):
        # TODO figure out what to do here, maybe just start with the end points and the mid point?
        # Could always sample more
        pass
        # # need to refit the space after collecting a new model
        # cov_factor = None

        # w = flatten([param.detach().cpu() for param in self.model.parameters()])
        
        # # first moment
        # self.mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        # self.mean.add_(w / (self.n_models.item() + 1.0))

        # # second moment
        # self.sq_mean.mul_(self.n_models.item() / (self.n_models.item() + 1.0))
        # self.sq_mean.add_(w ** 2 / (self.n_models.item() + 1.0))

        # dev_vector = w - self.mean

        # self.subspace.collect_vector(dev_vector)
        # def collect_vector(self, vector):
        #     if self.rank.item() + 1 > self.max_rank:
        #         self.cov_mat_sqrt = self.cov_mat_sqrt[1:, :]
        #     self.cov_mat_sqrt = torch.cat((self.cov_mat_sqrt, vector.view(1, -1)), dim=0)
        #     self.rank = torch.min(self.rank + 1, torch.as_tensor(self.max_rank)).view(-1)
        #     self.n_models.add_(1)
        
        # return self.mean.clone(), torch.clamp(self.sq_mean - self.mean ** 2, self.var_clamp).clone(), cov_factor.clone()
    
    def fit_and_eval_posterior(self):
        mean, var, cov_factor = self.get_subspace_mean_covar()
        
        vi_model = VIModel(
            subspace=SubspaceModel(mean.to(self.device), cov_factor.to(self.device)),
            init_inv_softplus_sigma=math.log(math.exp(self.init_std) - 1.0),
            prior_log_sigma=math.log(self.prior_std),
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
        self.optimizer = torch.optim.SGD([param for param in vi_model.parameters()],
                                    lr=self.learning_rate,
                                    momentum=0.9)
        columns = ['ep', 'acc', 'loss', 'kl', 'nll', 'sigma_1', 'time']

        epoch = 0
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
        eval_model = model_cfg.base(num_classes=num_classes,
                                    *model_cfg.args,
                                    **model_cfg.kwargs)
        eval_model.to(args.device)

        num_samples = args.num_samples

        ens_predictions = np.zeros((len(loaders['test'].dataset), num_classes))
        targets = np.zeros(len(loaders['test'].dataset))

        columns = ['iter ens', 'acc', 'nll']

        for i in range(num_samples):
            with torch.no_grad():
                w = vi_model.sample()
                offset = 0
                for param in eval_model.parameters():
                    param.data.copy_(w[offset:offset + param.numel()].view(
                        param.size()).to(self.device))
                    offset += param.numel()

            utils.bn_update(loaders['train'],
                            eval_model,
                            subset=args.bn_subset)

            pred_res = utils.predict(loaders['test'], eval_model)
            ens_predictions += pred_res['predictions']
            targets = pred_res['targets']

            values = [
                '%3d/%3d' % (i + 1, num_samples),
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

        ens_predictions /= num_samples
        ens_acc = np.mean(np.argmax(ens_predictions, axis=1) == targets)
        ens_nll = nll(ens_predictions, targets)
        
        return ens_acc, ens_nll
