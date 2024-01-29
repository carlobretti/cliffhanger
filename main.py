import math
import os
from functools import partial
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from base_run import BaseRun
from GTST_datamodule import GTSTDataModule
from GTSTDataset import GTST
from losses import BinaryFocalLossWithLogits
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score, Precision, Recall

# transformer code from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for l in self.layers:
            x = l(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for l in self.layers:
            _, attn_map = l.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = l(x)
        return attention_maps


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return x


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class SummPredictor(LightningModule):
    # def __init__(self, input_dim, model_dim, num_classes, num_heads, num_layers, lr, warmup, max_iters, dropout=0.0, input_dropout=0.0):
    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads,
        num_layers,
        lr,
        warmup,
        max_iters,
        focal_alpha,
        focal_gamma,
        dropout=0.0,
        input_dropout=0.0,
    ):
        """
        Inputs:
            input_dim - Hidden dimensionality of the input
            model_dim - Hidden dimensionality to use inside the Transformer
            num_classes - Number of classes to predict per sequence element
            num_heads - Number of heads to use in the Multi-Head Attention blocks
            num_layers - Number of encoder blocks to use.
            lr - Learning rate in the optimizer
            warmup - Number of warmup steps. Usually between 50 and 500
            max_iters - Number of maximum iterations the model is trained for. This is needed for the CosineWarmup scheduler
            dropout - Dropout to apply inside the model
            input_dropout - Dropout to apply on the input features
        """
        super().__init__()
        self.save_hyperparameters()
        self._create_model()

        # # using BCEWithLogitsLoss as it is more numerically stable than plain BCELoss
        # # pos_weight to make positive examples count 50 times as much towards the loss
        # self.loss_module = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(loss_pos_weight))

        self.loss_module = BinaryFocalLossWithLogits(
            alpha=focal_alpha, gamma=focal_gamma, reduction="mean"
        )

        self.acc_module = Accuracy(task="binary", threshold=0.5)
        self.rec_module = Recall(task="binary", threshold=0.5)
        self.prec_module = Precision(task="binary", threshold=0.5)
        self.f1_module = F1Score(task="binary", threshold=0.5)

    def _create_model(self):
        # Input dim -> Model dim
        self.input_net = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim),
        )

        # # Leave input unchanged
        # self.input_net = nn.Identity()

        # Positional encoding for sequences
        self.positional_encoding = PositionalEncoding(d_model=self.hparams.model_dim)
        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim=self.hparams.model_dim,
            dim_feedforward=2 * self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
        )

        # # Output classifier per sequence lement
        # self.output_net = nn.Sequential(
        #     nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
        #     nn.LayerNorm(self.hparams.model_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(self.hparams.dropout),
        #     nn.Linear(self.hparams.model_dim, self.hparams.num_classes)
        # )

        # Output classifier per sequence element
        self.output_net = nn.Sequential(
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
            nn.LayerNorm(self.hparams.model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.hparams.dropout),
            nn.Linear(self.hparams.model_dim, self.hparams.num_classes),
        )

    def forward(self, x, mask=None, add_positional_encoding=True):
        """
        Inputs:
            x - Input features of shape [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs (optional)
            add_positional_encoding - If True, we add the positional encoding to the input.
                                      Might not be desired for some tasks.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        x = self.transformer(x, mask=mask)
        out = self.output_net(x)
        return out

    def _calculate_loss(self, batch, mode="train"):
        """
        Calculate loss for a given batch and mode.
        Additionally, compute and log metrics for the batch at the frame level.
        """
        x = batch["features"]
        y = batch["labels"]
        metadata = batch["metadata"]
        logits = self.forward(x)
        y = y.unsqueeze(-1)
        loss = self.loss_module(input=logits, target=y.float())

        batch_size = x.size()[0]
        self.log(f"{mode}/loss", loss, batch_size=batch_size)

        if mode in ["train", "val", "test"]:
            preds = nn.Sigmoid()(logits)
            # n_frames = metadata["frame_level_gt"]["mask"].size()[1]
            frame_level_preds = torch.zeros_like(
                metadata["frame_level_gt"]["mask"].squeeze(), dtype=torch.float
            )
            if metadata["as_shots"] == True:
                # "inflate" predictions at a shot-level (masked) backed to the unmasked shot-level
                shots_unmasked = torch.zeros_like(
                    metadata["shot_level_mask"].squeeze(), dtype=torch.float
                )
                shots_unmasked[
                    metadata["shot_level_mask"].squeeze() == 1
                ] = preds.squeeze()

                # copy preds at shot level to frame level
                for (shot_start, shot_end), pred in zip(
                    metadata["shot_to_frame_map"].squeeze(), shots_unmasked
                ):
                    frame_level_preds[shot_start : shot_end + 1] = pred
            elif metadata["as_shots"] == False:
                # "inflate" predictions at a clip-level (masked) backed to the unmasked clip-level
                clip_unmasked = torch.zeros_like(
                    metadata["clip_level_mask"].squeeze(), dtype=torch.float
                )
                clip_unmasked[
                    metadata["clip_level_mask"].squeeze() == 1
                ] = preds.squeeze()

                for i, pred in enumerate(clip_unmasked):
                    start_frame = i * metadata["window_size"].squeeze()
                    end_frame = start_frame + metadata["window_size"].squeeze()
                    # no need to add 1 here as window size is the length of the clip as a whole, not end point as in shots
                    frame_level_preds[start_frame:end_frame] = pred

            # mask back preds at a frame level
            preds_frame = frame_level_preds[
                metadata["frame_level_gt"]["mask"].squeeze()
            ]

            y_frame = metadata["frame_level_gt"][metadata["ground_truth"][0]].squeeze()

            acc = self.acc_module(preds_frame, y_frame)
            rec = self.rec_module(preds_frame, y_frame)
            prec = self.prec_module(preds_frame, y_frame)
            f1 = self.f1_module(preds_frame, y_frame)

            self.log(f"{mode}/frame_acc", acc, batch_size=batch_size)
            self.log(f"{mode}/frame_rec", rec, batch_size=batch_size)
            self.log(f"{mode}/frame_prec", prec, batch_size=batch_size)
            self.log(f"{mode}/frame_f1", f1, batch_size=batch_size)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        # print(batch['filename'])
        _ = self._calculate_loss(batch, mode="test")

    def predict_step(self, batch, batch_idx, dataloader_idx):
        # print(batch['filename'])
        x = batch["features"]
        y = batch["labels"]
        logits = self.forward(x)
        preds = nn.Sigmoid()(logits)

        fig, ax = plt.subplots()
        ax.plot(preds.cpu().squeeze())
        ax.plot(y.cpu().squeeze(), "g")
        dataloader_idx_lookup = {0: "val", 1: "test"}
        self.logger.log_image(
            f"preds/{dataloader_idx_lookup[dataloader_idx]}/{batch['filename'][0]}",
            [wandb.Image(fig)],
        )
        plt.close()
        return preds

    @torch.no_grad()
    def get_attention_maps(self, x, mask=None, add_positional_encoding=True):
        """
        Function for extracting the attention matrices of the whole Transformer for a single batch.
        Input arguments same as the forward pass.
        """
        x = self.input_net(x)
        if add_positional_encoding:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x, mask=mask)
        return attention_maps

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)

        # Apply lr scheduler per step
        lr_scheduler = CosineWarmupScheduler(
            optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]


@hydra.main(config_path="config", config_name="transformer_config", version_base="1.1")
def func(cfg: DictConfig):
    seed_everything(cfg.general.seed)

    # datamodule parameters

    # hacky way of doing this since dataloader is only created later in the GTSTKFoldDataLoader
    orig_cwd = hydra.utils.get_original_cwd()
    # root = f"{'/'.join(orig_cwd.split('/')[:-1])}"
    root = orig_cwd
    _tempdset = GTST(
        root=root,
        as_shots=cfg.datamodule.as_shots,
        as_semantic=cfg.datamodule.as_semantic,
        ground_truth=cfg.datamodule.ground_truth,
    )

    # baserun params
    export_path = f"./experiments"
    data_type = f"as_semantic{cfg.datamodule.as_semantic}.as_shots{cfg.datamodule.as_shots}.ground_truth{cfg.datamodule.ground_truth}"
    save_dir = f"{os.getcwd()}/{export_path}/logs/{cfg.general.model_name}/{data_type}"
    # trainer parameters
    module = partial(SummPredictor)
    logger = partial(WandbLogger)

    callbacks_with_args = {
        "funcs": {
            "learning_rate_monitor": partial(LearningRateMonitor),
            "model_checkpoint": partial(ModelCheckpoint),
        },
        "args": {
            "learning_rate_monitor": {"logging_interval": "epoch"},
            "model_checkpoint": {
                "dirpath": save_dir + "/checkpoints",
                "save_weights_only": cfg.trainer.model_checkpoint_save_weights_only,
                "mode": cfg.trainer.model_checkpoint_mode,
                "monitor": cfg.trainer.model_checkpoint_monitor,
            },
        },
    }

    if cfg.trainer.early_stopping:
        callbacks_with_args["funcs"]["early_stopping"] = partial(EarlyStopping)
        callbacks_with_args["args"]["early_stopping"] = {
            "monitor": cfg.trainer.early_stopping_monitor,
            "mode": cfg.trainer.early_stopping_mode,
            "min_delta": cfg.trainer.early_stopping_min_delta,
            "patience": cfg.trainer.early_stopping_patience,
            "verbose": True,
        }

    max_iters = cfg.trainer.max_epochs * int(
        (1 - cfg.datamodule.test_split_ratio) * len(_tempdset)
    )

    # model parameters
    # input dim depend on data type
    input_dim = _tempdset[0]["features"].shape[1]

    datamodule = GTSTDataModule(
        root=root,
        batch_size=cfg.datamodule.batch_size,
        window_size=cfg.datamodule.window_size,
        test_split_ratio=cfg.datamodule.test_split_ratio,
        split_seed=cfg.datamodule.split_seed,
        as_shots=cfg.datamodule.as_shots,
        as_semantic=cfg.datamodule.as_semantic,
        ground_truth=cfg.datamodule.ground_truth,
    )

    # save hydra configs in dict format
    _cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    run = BaseRun(
        model_name=cfg.general.model_name,
        data_type=data_type,
        export_path=export_path,
        _cfg=_cfg,
    )

    run.setup_modules(module=module, datamodule=datamodule)

    run.setup_module_args(
        input_dim=input_dim,
        model_dim=cfg.module.model_dim,
        num_heads=cfg.module.num_heads,
        num_layers=cfg.module.num_layers,
        num_classes=cfg.module.num_classes,
        dropout=cfg.module.dropout,
        input_dropout=cfg.module.input_dropout,
        lr=cfg.module.lr,
        warmup=cfg.module.warmup,
        max_iters=max_iters,
        focal_alpha=cfg.module.focal_alpha,
        focal_gamma=cfg.module.focal_gamma,
    )

    run.setup_trainer_args(
        max_epochs=cfg.trainer.max_epochs,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        devices=cfg.trainer.devices,
        accelerator=cfg.trainer.accelerator,
        logger=logger,
        callbacks=callbacks_with_args,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
    )

    run.run()


if __name__ == "__main__":
    func()
