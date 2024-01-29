from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, Subset

from GTSTDataset import GTST


@dataclass
class GTSTDataModule(LightningDataModule):
    root: str = "."
    train_dataset: Optional[Dataset] = None
    val_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None
    test_split_ratio: Optional[float] = 0.2
    batch_size: Optional[int] = 1
    name: Optional[str] = "gtst"
    num_workers: int = 10
    window_size: int = 64
    as_shots: bool = False
    as_semantic: bool = False
    ground_truth: str = "only_preview"
    split_seed: int = 42
    _has_setup_fit: bool = False

    def prepare_data(self) -> None:

        pass

    def setup(self, stage: Optional[str] = None) -> None:
        if self._has_setup_fit:
            return
        # load the data
        dataset = GTST(
            root=self.root,
            window_size=self.window_size,
            as_shots=self.as_shots,
            as_semantic=self.as_semantic,
            ground_truth=self.ground_truth,
        )
        test_size = int(len(dataset) * self.test_split_ratio)
        # validation set the same size as the test set
        train_size = len(dataset) - 2 * test_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset,
            [train_size, test_size, test_size],
            generator=torch.Generator().manual_seed(self.split_seed),
        )
        # to ensure setup is only called once
        self._has_setup_fit = True

    def train_dataloader(self) -> DataLoader:
        # print('called train_dataloader')
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self) -> DataLoader:
        # print('called val_dataloader')
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self) -> DataLoader:
        # print('called test_dataloader')
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def predict_dataloader(self) -> DataLoader:
        # print('called predict_dataloader')
        loaders = [
            DataLoader(
                dataset=self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
            ),
            DataLoader(
                dataset=self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                drop_last=False,
            ),
        ]
        return loaders

    def __post_init__(self):
        super().__init__()
        self.save_hyperparameters()

