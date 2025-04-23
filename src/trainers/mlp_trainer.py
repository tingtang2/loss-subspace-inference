import logging
import random

import torch
import torchvision
import torchvision.transforms as transforms
from models.mlp import NN, NonLinearSubspaceNN, SubspaceNN
from torch import nn
from torch.distributions.exponential import Exponential
from torch.utils.data import DataLoader
from tqdm import trange

from trainers.base_trainer import BaseTrainer


class MLPTrainer(BaseTrainer):

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(loader):
            self.optimizer.zero_grad()

            reshaped_x = x.reshape(x.size(0), 784)

            y_hat = self.model(reshaped_x.to(self.device))
            loss = self.criterion(y_hat, y.to(self.device))

            loss.backward()
            running_loss += loss.item()

            self.optimizer.step()

        return running_loss / len(loader.dataset)

    def eval(self, loader: DataLoader):
        num_right = 0
        running_loss = 0.0

        self.model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(loader):
                reshaped_x = x.reshape(x.size(0), 784)
                y_hat = self.model(reshaped_x.to(self.device))
                num_right += torch.sum(
                    y.to(self.device) == torch.argmax(
                        y_hat, dim=-1)).detach().cpu().item()

                running_loss += self.criterion(y_hat, y.to(self.device)).item()

        return num_right / len(loader.dataset), (running_loss /
                                                 len(loader.dataset))

    def run_experiment(self, iter: int):
        self.create_dataloaders()

        if 'subspace' in self.name and 'nonlinear' in self.name:
            print("Fitting nonlinear subspace to loss landscape")
            self.model = NonLinearSubspaceNN(input_dim=self.data_dim,
                                             hidden_dim=self.hidden_size,
                                             out_dim=self.out_dim,
                                             dropout_prob=self.dropout_prob,
                                             seed=self.seed).to(self.device)

        elif 'subspace' in self.name:
            print("Fitting linear subspace to loss landscape")
            if 'simplex' in self.name:
                num_weights = 3
            else:
                num_weights = 2
            self.model = SubspaceNN(input_dim=self.data_dim,
                                    hidden_dim=self.hidden_size,
                                    out_dim=self.out_dim,
                                    dropout_prob=self.dropout_prob,
                                    seed=self.seed,
                                    num_weights=num_weights).to(self.device)

        else:
            self.model = NN(input_dim=self.data_dim,
                            hidden_dim=self.hidden_size,
                            out_dim=self.out_dim,
                            dropout_prob=self.dropout_prob).to(self.device)

        self.optimizer = self.optimizer_type(self.model.parameters(),
                                             lr=self.learning_rate)

        training_loss = []
        val_loss = []
        training_accuracy = []
        val_accuracy = []
        cos_sims = []
        l2s = []

        best_val_loss = 1e+5
        early_stopping_counter = 0

        for i in trange(1, self.epochs + 1):
            training_loss.append(self.train_epoch(self.train_loader))
            if 'subspace' in self.name:
                if self.val_midpoint_only:
                    idx = 0
                else:
                    idx = 1

                training_accuracy.append(self.eval(self.train_loader)[0][idx])
                accuracy, loss, cos_sim, l2 = self.eval(self.valid_loader)

                l2s.append(l2)
                cos_sims.append(cos_sim)

                if self.val_midpoint_only:
                    logging.info(
                        f'epoch: {i} training loss: {training_loss[-1]:.3f} val loss:{loss[0]:.3f} training accuracy: {training_accuracy[-1]:.3f} val acc: {accuracy[0]:.3f}, patience: {early_stopping_counter} '
                        f'cos sim: {cos_sim:.3E}, l2: {l2:.3f}')
                else:
                    logging.info(
                        f'epoch: {i} training loss: {training_loss[-1]:.3f} val loss alpha 0: {loss[0]:.3f} val loss alpha 0.5: {loss[1]:.3f} val loss alpha 1: {loss[-1]:.3f} training accuracy: {training_accuracy[-1]:.3f} '
                        f'val acc 0: {accuracy[0]:.3f}, val acc 0.5: {accuracy[1]:.3f}, val acc 1: {accuracy[-1]:.3f}, patience: {early_stopping_counter} cos sim: {cos_sim:.3E}, l2: {l2:.3f}'
                    )

                loss = loss[idx]
            else:
                training_accuracy.append(self.eval(self.train_loader)[0])
                acc, loss = self.eval(self.valid_loader)
                val_accuracy.append(acc)
                val_loss.append(loss)

                logging.info(
                    f'epoch: {i} training loss: {training_loss[-1]:.3f} val loss:{val_loss[-1]:.3f} training accuracy: {training_accuracy[-1]:.3f} val acc: {val_accuracy[-1]:.3f}, patience: {early_stopping_counter}'
                )

            if loss < best_val_loss:
                self.save_model(f'{self.name}_{iter}')
                early_stopping_counter = 0
                best_val_loss = loss
            else:
                early_stopping_counter += 1

            if early_stopping_counter == self.early_stopping_threshold:
                break


class FashionMNISTMLPTrainer(MLPTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.name = f'vanilla_mlp_seed_{self.seed}'
        self.early_stopping_threshold = 10

        self.data_dim = 784
        self.out_dim = 10

    def create_dataloaders(self):
        transform = transforms.Compose([transforms.ToTensor()])
        FashionMNIST_data_train = torchvision.datasets.FashionMNIST(
            self.data_dir, train=True, transform=transform, download=True)

        train_set, val_set = torch.utils.data.random_split(
            FashionMNIST_data_train, [50000, 10000])
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(
            val_set, batch_size=len(val_set), shuffle=False)


class SubspaceMLPTrainer(MLPTrainer):

    def get_weight(self, m, i):
        if i == 0:
            return m.weight
        return getattr(m, f'weight_{i}')

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(loader):
            alpha = torch.rand(1, device=self.device)
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    # add attribute for weight dimensionality and subspace dimensionality
                    setattr(m, f'alpha', alpha)

            self.optimizer.zero_grad()

            reshaped_x = x.reshape(x.size(0), 784)

            y_hat = self.model(reshaped_x.to(self.device))
            loss = self.criterion(y_hat, y.to(self.device))

            # regularization
            num = 0.0
            norm = 0.0
            norm1 = 0.0
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Linear) and 'parameterization' not in name:
                    vi = self.get_weight(m, 0)
                    vj = self.get_weight(m, 1)

                    num += (vi * vj).sum()
                    norm += vi.pow(2).sum()
                    norm1 += vj.pow(2).sum()

            loss += self.beta * (num.pow(2) / (norm * norm1))

            loss.backward()
            running_loss += loss.item()

            self.optimizer.step()

        return running_loss / len(loader.dataset)

    def eval(self, loader: DataLoader):
        running_losses = [0.0, 0.0, 0.0]
        alphas = [0.0, 0.5, 1.0]
        nums_right = [0, 0, 0]

        if self.val_midpoint_only:
            alphas = [0.5]

        self.model.eval()

        for i, alpha in enumerate(alphas):
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    setattr(m, f'alpha',
                            torch.tensor([alpha], device=self.device))

            with torch.no_grad():
                for j, (x, y) in enumerate(loader):
                    reshaped_x = x.reshape(x.size(0), 784)
                    y_hat = self.model(reshaped_x.to(self.device))
                    nums_right[i] += torch.sum(
                        y.to(self.device) == torch.argmax(
                            y_hat, dim=-1)).detach().cpu().item()

                    running_losses[i] += self.criterion(
                        y_hat, y.to(self.device)).item()

        # compute l2 and cos sim
        num = 0.0
        norm = 0.0
        norm1 = 0.0

        total_l2 = 0.0

        for name, m in self.model.named_modules():
            if (isinstance(m, nn.Linear) or isinstance(
                    m, nn.Embedding)) and 'parameterization' not in name:

                vi = self.get_weight(m, 0)
                vj = self.get_weight(m, 1)

                num += (vi * vj).sum()
                norm += vi.pow(2).sum()
                norm1 += vj.pow(2).sum()

                total_l2 += (vi - vj).pow(2).sum()

        total_cosim = num.pow(2) / (norm * norm1)
        total_l2 = total_l2.sqrt()

        return [num / len(loader.dataset) for num in nums_right
                ], [loss / len(loader.dataset) for loss in running_losses
                    ], total_cosim.item(), total_l2.item()


class FashionMNISTSubspaceMLPTrainer(SubspaceMLPTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.name = 'subspace_vanilla_mlp'
        self.early_stopping_threshold = 10

        self.data_dim = 784
        self.out_dim = 10

    def create_dataloaders(self):
        transform = transforms.Compose([transforms.ToTensor()])
        FashionMNIST_data_train = torchvision.datasets.FashionMNIST(
            self.data_dir, train=True, transform=transform, download=False)

        train_set, val_set = torch.utils.data.random_split(
            FashionMNIST_data_train, [50000, 10000])
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(
            val_set, batch_size=len(val_set), shuffle=False)


class NonLinearSubspaceMLPTrainer(MLPTrainer):

    def get_weight(self, m, i):
        return m.line.forward(
            torch.tensor([i], dtype=torch.float32, device=self.device))

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(loader):
            alpha = torch.rand(1, device=self.device)
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Linear) and 'parameterization' not in name:
                    # add attribute for weight dimensionality and subspace dimensionality
                    setattr(m, f'alpha', alpha)

            self.optimizer.zero_grad()

            reshaped_x = x.reshape(x.size(0), 784)

            y_hat = self.model(reshaped_x.to(self.device))
            loss = self.criterion(y_hat, y.to(self.device))

            # regularization
            num = 0.0
            norm = 0.0
            norm1 = 0.0
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Linear) and 'parameterization' not in name:
                    vi = self.get_weight(m, 0)
                    vj = self.get_weight(m, 1)

                    num += (vi * vj).sum()
                    norm += vi.pow(2).sum()
                    norm1 += vj.pow(2).sum()

            loss += self.beta * (num.pow(2) / (norm * norm1))

            loss.backward()
            running_loss += loss.item()

            self.optimizer.step()

        return running_loss / len(loader.dataset)

    def eval(self, loader: DataLoader):
        running_losses = [0.0, 0.0, 0.0]
        alphas = [0.0, 0.5, 1.0]
        nums_right = [0, 0, 0]

        if self.val_midpoint_only:
            alphas = [0.5]

        self.model.eval()

        for i, alpha in enumerate(alphas):
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    setattr(m, f'alpha',
                            torch.tensor([alpha], device=self.device))

            with torch.no_grad():
                for j, (x, y) in enumerate(loader):
                    reshaped_x = x.reshape(x.size(0), 784)
                    y_hat = self.model(reshaped_x.to(self.device))
                    nums_right[i] += torch.sum(
                        y.to(self.device) == torch.argmax(
                            y_hat, dim=-1)).detach().cpu().item()

                    running_losses[i] += self.criterion(
                        y_hat, y.to(self.device)).item()

        # compute l2 and cos sim
        num = 0.0
        norm = 0.0
        norm1 = 0.0

        total_l2 = 0.0

        for name, m in self.model.named_modules():
            if (isinstance(m, nn.Linear) or isinstance(
                    m, nn.Embedding)) and 'parameterization' not in name:
                vi = self.get_weight(m, 0)
                vj = self.get_weight(m, 1)

                num += (vi * vj).sum()
                norm += vi.pow(2).sum()
                norm1 += vj.pow(2).sum()

                total_l2 += (vi - vj).pow(2).sum()

        total_cosim = num.pow(2) / (norm * norm1)
        total_l2 = total_l2.sqrt()

        return [num / len(loader.dataset) for num in nums_right
                ], [loss / len(loader.dataset) for loss in running_losses
                    ], total_cosim.item(), total_l2.item()


class FashionMNISTNonLinearSubspaceMLPTrainer(NonLinearSubspaceMLPTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.name = f'nonlinear_subspace_vanilla_mlp_seed_{self.seed}_beta_{self.beta}_tanh'
        self.early_stopping_threshold = 10

        self.data_dim = 784
        self.out_dim = 10

    def create_dataloaders(self):
        transform = transforms.Compose([transforms.ToTensor()])
        FashionMNIST_data_train = torchvision.datasets.FashionMNIST(
            self.data_dir, train=True, transform=transform, download=False)

        train_set, val_set = torch.utils.data.random_split(
            FashionMNIST_data_train, [50000, 10000])
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(
            val_set, batch_size=len(val_set), shuffle=False)


class SimplexSubspaceMLPTrainer(MLPTrainer):

    def get_weight(self, m, i):
        if i == 0:
            return m.weight
        return getattr(m, f'weight_{i}')

    def train_epoch(self, loader: DataLoader):
        self.model.train()
        running_loss = 0.0

        for i, (x, y) in enumerate(loader):
            dist = Exponential(rate=1)
            Z = dist.sample(sample_shape=(3, ))
            Z = Z / Z.sum()
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    # add attribute for weight dimensionality and subspace dimensionality
                    for i in range(1, 3):
                        setattr(m, f't{i}', Z[i])

            self.optimizer.zero_grad()

            reshaped_x = x.reshape(x.size(0), 784)

            y_hat = self.model(reshaped_x.to(self.device))
            loss = self.criterion(y_hat, y.to(self.device))

            # regularization
            out = random.sample([i for i in range(3)], 2)
            i, j = out[0], out[1]
            num = 0.0
            norm = 0.0
            norm1 = 0.0
            for name, m in self.model.named_modules():
                if isinstance(m, nn.Linear) and 'parameterization' not in name:
                    vi = self.get_weight(m, i)
                    vj = self.get_weight(m, j)

                    num += (vi * vj).sum()
                    norm += vi.pow(2).sum()
                    norm1 += vj.pow(2).sum()

            loss += self.beta * (num.pow(2) / (norm * norm1))

            loss.backward()
            running_loss += loss.item()

            self.optimizer.step()

        return running_loss / len(loader.dataset)

    def eval(self, loader: DataLoader):
        running_losses = [0.0, 0.0, 0.0]
        alphas = [0.0, 0.5, 1.0]
        nums_right = [0, 0, 0]

        if self.val_midpoint_only:
            alphas = [0.5]

        self.model.eval()

        for i, alpha in enumerate(alphas):
            for m in self.model.modules():
                if isinstance(m, nn.Linear):
                    for k in range(3):
                        setattr(m, f't{k}', 1.0 / 3)

            with torch.no_grad():
                for j, (x, y) in enumerate(loader):
                    reshaped_x = x.reshape(x.size(0), 784)
                    y_hat = self.model(reshaped_x.to(self.device))
                    nums_right[i] += torch.sum(
                        y.to(self.device) == torch.argmax(
                            y_hat, dim=-1)).detach().cpu().item()

                    running_losses[i] += self.criterion(
                        y_hat, y.to(self.device)).item()

        # compute l2 and cos sim
        num = 0.0
        norm = 0.0
        norm1 = 0.0

        total_l2 = 0.0

        for name, m in self.model.named_modules():
            if (isinstance(m, nn.Linear) or isinstance(
                    m, nn.Embedding)) and 'parameterization' not in name:

                vi = self.get_weight(m, 0)
                vj = self.get_weight(m, 1)

                num += (vi * vj).sum()
                norm += vi.pow(2).sum()
                norm1 += vj.pow(2).sum()

                total_l2 += (vi - vj).pow(2).sum()

        total_cosim = num.pow(2) / (norm * norm1)
        total_l2 = total_l2.sqrt()

        return [num / len(loader.dataset) for num in nums_right
                ], [loss / len(loader.dataset) for loss in running_losses
                    ], total_cosim.item(), total_l2.item()


class FashionMNISTSimplexSubspaceMLPTrainer(SimplexSubspaceMLPTrainer):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.name = 'simplex_subspace_vanilla_mlp'
        self.early_stopping_threshold = 10

        self.data_dim = 784
        self.out_dim = 10

    def create_dataloaders(self):
        transform = transforms.Compose([transforms.ToTensor()])
        FashionMNIST_data_train = torchvision.datasets.FashionMNIST(
            self.data_dir, train=True, transform=transform, download=False)

        train_set, val_set = torch.utils.data.random_split(
            FashionMNIST_data_train, [50000, 10000])
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True)
        self.valid_loader = torch.utils.data.DataLoader(
            val_set, batch_size=len(val_set), shuffle=False)
