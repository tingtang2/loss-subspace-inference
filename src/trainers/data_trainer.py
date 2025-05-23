import torch
import torchvision
import torchvision.transforms as transforms

from trainers.base_trainer import BaseTrainer


class FashionMNISTTrainer(BaseTrainer):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

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
    
    def create_testloader(self):
        transform = transforms.Compose([transforms.ToTensor()])
        test_set = torchvision.datasets.FashionMNIST(
            self.data_dir, train=False, transform=transform, download=False)
        
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=len(test_set), shuffle=False)
