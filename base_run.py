# %%
import os
import os.path as osp
import shutil
from copy import deepcopy
from glob import glob
from pathlib import Path
from typing import Dict, Type

import torch
import wandb
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import CustomWriter


# %%
class BaseRun:
    def __init__(
        self, model_name: str, data_type: str, export_path: str, _cfg: Dict
    ) -> None:
        self.export_path = export_path
        self.model_name = model_name
        self.data_type = data_type
        self._cfg = _cfg

    def setup_modules(
        self, module: Type[LightningModule], datamodule: Type[LightningDataModule]
    ):
        self.module = module
        self.datamodule = datamodule

    def setup_trainer_args(self, **kwargs):
        self.trainer_args = kwargs

    def setup_module_args(self, **kwargs):
        self.module_args = kwargs

    def run(self):
        self.datamodule.prepare_data()
        self.datamodule.setup()
        wandb.finish()

        logger_part = self.trainer_args["logger"]
        callbacks = deepcopy(self.trainer_args["callbacks"])

        project_name = "trailerness"
        save_dir = (
            f"{os.getcwd()}/{self.export_path}/logs/{self.model_name}/{self.data_type}"
        )
        os.makedirs(save_dir)

        print("Working directory : {}".format(os.getcwd()))

        self.trainer_args["logger"] = logger_part(
            project=project_name,
            save_dir=save_dir,
            name=f"{self.model_name}",
            settings=wandb.Settings(start_method="fork"),
            reinit=True,
        )

        # save hydra config file
        self.trainer_args["logger"].experiment.config.update(self._cfg)

        # extra callbacks are called with their respective params
        self.trainer_args["callbacks"] = [
            callbacks["funcs"][name](**callbacks["args"][name])
            for name in callbacks["funcs"].keys()
        ]

        # save predictions
        writer_callback = CustomWriter(output_dir=save_dir, write_interval="batch")
        self.trainer_args["callbacks"].append(writer_callback)

        trainer = Trainer(**self.trainer_args)
        trainer.logger._default_hp_metric = None

        self.module = self.module(**self.module_args)

        print(f"TRAINING")
        trainer.fit(self.module, datamodule=self.datamodule)

        print("VALIDATE AFTER TRAINING WITH BEST MODEL")
        trainer.validate(
            ckpt_path=trainer.checkpoint_callback.best_model_path,
            datamodule=self.datamodule,
        )
        print(f"TESTING")
        trainer.test(
            ckpt_path=trainer.checkpoint_callback.best_model_path,
            datamodule=self.datamodule,
        )
        trainer.predict(
            ckpt_path=trainer.checkpoint_callback.best_model_path,
            datamodule=self.datamodule,
        )
        wandb.finish()
