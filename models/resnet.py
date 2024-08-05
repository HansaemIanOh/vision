import os
import sys
import math
import numpy as np
from typing import *
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision.utils as vutils
from . import Custumnn
from safetensors.torch import save_file, load_file

class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        stride: int = 1,
        act: Union[nn.Module, Callable] = nn.ReLU(),
        downsample: Optional[nn.Module] = None, 
    ) -> None:
        super().__init__()
        """conv -> norm -> act -> conv -> norm -> skip"""
        self.act = act
        self.downsample = downsample
        self.block = nn.ModuleList()
        self.block.append(nn.Sequential(
            nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            self.act,
            nn.Conv2d(out_features, out_features, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_features),
        ))
    def forward(self, x : Tensor) -> Tensor:
        h = x
        for module in self.block:
            h = module(h)
        if self.downsample is not None:
            x = self.downsample(x)
        return self.act(x + h)

class BottleNeck(nn.Module):
    expansion: int = 4
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        stride: int = 1,
        act: Union[nn.Module, Callable] = nn.ReLU(),
        downsample: Optional[nn.Module] = None, 
    ) -> None:
        super().__init__()
        """conv1x1 -> norm -> act -> conv3x3 -> norm -> act -> conv1x1 -> norm -> skip"""
        width = int(out_features / self.expansion)
        self.act = act
        self.downsample = downsample
        self.block = nn.ModuleList()
        self.block.append(nn.Sequential(
            nn.Conv2d(in_features, width, kernel_size=(1, 1), stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(width),
            self.act,
            nn.Conv2d(width, width, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width),
            self.act,
            nn.Conv2d(width, out_features, kernel_size=(1, 1), stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_features),
        ))

    def forward(self, x : Tensor) -> Tensor:
        h = x
        for module in self.block:
            h = module(h)
        if self.downsample is not None:
            x = self.downsample(x)
        return self.act(x + h)

class ResNetModel(pl.LightningModule):
    def __init__(
        self, 
        config, 
        **kwargs
    ) -> None:
        super().__init__()
        self.config = config
        self.h_dims = config['h_dims'] # [3, 64, 64, 128, 256, 512]
        self.block_depth = config['block_depth'] # [0, 2, 2, 2, 2] 0 is not used.
        self.patch_size = config['patch_size']
        self.dropout = config['dropout']
        self.num_classes = config['num_classes']
        if self.num_classes==0:
            sys.exit()
        blockinfo = {
            'BasicBlock':BasicBlock,
            'BottleNeck':BottleNeck
        }
        self.block = blockinfo[config['block']]
        self.act = nn.ReLU()
        num_channels = len(self.h_dims)
        self.conv1 = nn.Conv2d(self.h_dims[0], self.h_dims[1], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.h_dims[1])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        f_res = self.patch_size
        strides = [0, 1, 2, 2, 2]
        self.layers = nn.ModuleList()
        for index in range(1, num_channels-1):
            in_channels = self.h_dims[index]
            out_channels = self.h_dims[index+1]
            self.layers.append(
                self._make_layer(self.block, in_channels, out_channels, self.block_depth[index], stride=strides[index])
            )
            f_res /= 2
        self.fc = nn.Linear(512, self.num_classes)      
        self.train_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes) 
        self.train_losses = []
        self.val_losses = [] 
    def _make_layer(
        self, 
        block: Type[Union[BasicBlock, BottleNeck]],
        in_channels: int,
        out_channels: int,
        block_depth: int,
        stride: int = 1,
    ) -> nn.ModuleList:

        downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

        layers = []
        layers.append(
            block(
                in_channels, out_channels, stride, self.act, downsample
            )
        )
        for _ in range(1, block_depth):
            layers.append(
                block(
                    out_channels,
                    out_channels,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        h = x
        h = self.conv1(h)
        h = self.bn1(h)
        h = self.act(h)
        h = self.maxpool(h)
        '''[B, 64, 56, 56]'''
        for module in self.layers:
            h = module(h)

        h = self.avgpool(h)
        h = torch.flatten(h, 1)
        h = self.fc(h)
        return h

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
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
        y_hat = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
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
        y_hat = self(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
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