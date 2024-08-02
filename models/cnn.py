import os
import numpy as np
from typing import *
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision.utils as vutils
from safetensors.torch import save_file, load_file

class CNNModel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        h_dims = config['h_dims'] # [3, 16, 32, 64, 128, 256]
        patch_size = config['patch_size']
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']

        num_channels = len(h_dims)
        # Downsample
        self.Downsampling = nn.ModuleList()
        
        f_res = patch_size # [224, 112, 56, 28, 14, 7]
        for index in range(0, num_channels-1):
            in_channels = h_dims[index]
            out_channels = h_dims[index+1]

            self.Downsampling.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))
            f_res /= 2

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.latent_res = int(f_res)
        self.latent_ch = int(out_channels)
        self.FCLayers = nn.ModuleList()
        self.FCLayers.append(
            nn.Sequential(
            nn.Linear(self.latent_ch, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, self.num_classes),
            )
        )
        
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)

        self.train_losses = []
        self.val_losses = []
    def forward(self, x):
        h = x
        for module in self.Downsampling:
            h = module(h)
        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        for module in self.FCLayers:
            h = module(h)
        return F.log_softmax(h, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)
        self.log('Train Loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('Train Acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.train_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        # 에포크 종료 시 평균 손실 계산 및 저장
        avg_loss = np.mean(self.train_losses)
        self.train_losses = []  # 리스트 초기화
        self.save_loss('train', avg_loss)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        self.log('VL', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('VA', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.val_losses.append(loss.item())

    def on_validation_epoch_end(self):
        avg_loss = np.mean(self.val_losses)
        self.val_losses = []
        self.save_loss('val', avg_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        self.log('TeL', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('TeA', acc, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = \
        torch.optim.Adam(
            self.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
            )

        try:
            if self.config['scheduler_gamma'] is not None:
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer = optimizer, gamma = self.config['scheduler_gamma'])
                return [optimizer], [scheduler]
        except:
            return optimizer

    def save_loss(self, phase, loss):
        save_path = os.path.join(self.logger.log_dir, f'{phase}_losses.npy')
        if self.trainer.is_global_zero:  # Preventing duplication for multi GPU environment
            try:
                losses = np.load(save_path)
                losses = np.append(losses, loss)
            except FileNotFoundError:
                losses = np.array([loss])
            np.save(save_path, losses)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config, *args, **kwargs):
        model = cls(config, *args, **kwargs)
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        return model