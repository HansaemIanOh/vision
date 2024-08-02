import os
from typing import Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder

class ImageFolderDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, patch_size: int = 224, batch_size: int = 32, num_workers: int = 4, 
                 val_split: float = 0.1, test_split: float = 0.1, 
                 use_manual_split: bool = False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.use_manual_split = use_manual_split

        self.transform = transforms.Compose([
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage: Optional[str] = None):
        if self.use_manual_split:
            self.full_dataset = ImageFolder(self.data_dir, transform=self.transform)
            self._split_dataset()
        else:
            self.train_dataset = ImageFolder(os.path.join(self.data_dir, 'train'), transform=self.transform)
            self.val_dataset = ImageFolder(os.path.join(self.data_dir, 'val'), transform=self.transform)
            self.test_dataset = ImageFolder(os.path.join(self.data_dir, 'test'), transform=self.transform)

    def _split_dataset(self):
        total_size = len(self.full_dataset)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - test_size - val_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

