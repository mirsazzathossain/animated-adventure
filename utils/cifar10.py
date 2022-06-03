from torchvision.datasets import CIFAR10
from utils.setup_configs import load_config
import pytorch_lightning as pl
from preprocessing.transforms import *
from torch.utils.data import DataLoader, random_split


config = load_config()

class Cifar10DataModule(pl.LightningDataModule):
    def __init__(
        self, data_dir: str = config[1]['exp_params']['path_dataset'], batch_size: int = config[1]['exp_params']['batch_size'], num_workers: int = config[1]['exp_params']['num_workers']
        ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_transforms = train_transforms 
        self.test_transforms = test_transforms
        self.num_workers = num_workers

    def prepare_data(self):
        """
        Download the data if it is not present.
        """
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """
        Assign train, test, validation datasets.
        """
        if stage == 'fit' or stage is None:
            cifar10_full = CIFAR10(self.data_dir, train=True, transform=self.train_transforms)
            self.cifar10_train, self.cifar10_val = random_split(cifar10_full, [40000, 10000])

        if stage == 'test' or stage is None:
            self.cifar10_test = CIFAR10(root=self.data_dir, train=False, transform=self.test_transforms)

    def train_dataloader(self):
        """
        Returns:
            train_loader: DataLoader for training.
        """
        return DataLoader(self.cifar10_train, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        """
        Returns:
            val_loader: DataLoader for validation.
        """
        return DataLoader(self.cifar10_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        """
        Returns:
            test_loader: DataLoader for testing.
        """
        return DataLoader(self.cifar10_test, batch_size=self.batch_size, num_workers=self.num_workers)
