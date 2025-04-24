import torch
import torchvision
import torchvision.transforms as transforms

from trainers.base_trainer import BaseTrainer


class FashionMNISTTrainer(BaseTrainer):

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
