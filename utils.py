import os
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import Any, List, Type
from pytorch_lightning.core import LightningModule


class CustomWriter(BasePredictionWriter):
    def __init__(self, output_dir: str, write_interval: str):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_batch_end(
        self, trainer, pl_module: Type[LightningModule], prediction: Any, batch_indices: List[int], batch: Any,
        batch_idx: int, dataloader_idx: int
    ):
        filename = batch['filename']
        # have to hardcode this, order is determined in GTST_datamodule
        dataloader_idx_lookup = {0:'val',1:'test'}
        # print(f'saving preds, {filename}, {prediction.size()}')
        torch.save(prediction, os.path.join(self.output_dir, f"{dataloader_idx_lookup[dataloader_idx]}_{filename[0]}.pt"))
