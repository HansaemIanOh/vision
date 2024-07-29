import os
from typing import *
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision.utils as vutils

class FLOWERCNNModel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        h_dims = config['h_dims'] # [3, 16, 32, 64, 128, 256]
        patch_size = config['patch_size']

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
            nn.Linear(256, 10),
            )
        )
        
        self.train_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
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
        self.log('TL', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('TA', acc, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        self.log('Val Loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('Val Acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(preds, y)
        self.log('Test Loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log('Test Acc', acc, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])

