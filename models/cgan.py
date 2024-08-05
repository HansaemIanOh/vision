import os
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

class GENERATOR(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.st_res = config['generator:st_res']
        self.h_dims = config['h_dim:generator']
        self.num_channels = len(self.h_dims)
        self.num_classes = config['num_classes']
        self.latent_dim = self.h_dims[0]

        # Embedding for the class labels
        self.label_emb = nn.Embedding(self.num_classes, self.latent_dim * self.st_res * self.st_res)

        f_res = 1
        # generator
        self.generator = nn.ModuleList()
        for index in range(0, self.num_channels-1):
            in_channels = self.h_dims[index]
            out_channels = self.h_dims[index+1]

            self.generator.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ))
            f_res *= 2
        
        # Final
        self.generator.append(
            nn.Sequential(
                Custumnn.Linear(out_channels, out_channels),
            ))

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        label_embeddings = self.label_emb(y).view(y.size(0), self.latent_dim, self.st_res, self.st_res)
        '''
        x : [B, latent_dim, st_res, st_res]
        y : [B, latent_dim, st_res, st_res]
        '''
        h = x 
        h = h + label_embeddings
        for module in self.generator:
            h = module(h)
        return h


class DISCRIMINATOR(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.st_res = config['generator:st_res']
        self.h_dims = config['h_dim:discriminator']
        self.patch_size = config['patch_size']
        self.num_channels = len(self.h_dims)
        self.num_classes = config['num_classes']
        

        f_res = self.patch_size
        # discriminator
        self.discriminator = nn.ModuleList()
        for index in range(0, self.num_channels-1):
            in_channels = self.h_dims[index] # if index != 0 else self.h_dims[index] + 1
            out_channels = self.h_dims[index+1]

            self.discriminator.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                ))
            f_res /= 2
        assert int(f_res) == f_res, f"f_res must be an integer, but got {f_res}"
        # Final
        self.final = nn.Sequential(
            Custumnn.Linear(out_channels * int(f_res)**2, 1),
        )

        # Embedding for the class labels
        self.label_emb = nn.Embedding(self.num_classes, out_channels * int(f_res)**2)

    def forward(self, x, y):
        '''
        x : [B, 3, patch_size, patch_size]
        '''
        label_embeddings = self.label_emb(y)
        
        h = x
        for module in self.discriminator:
            h = module(h)
        h = torch.flatten(h, 1)
        h = h + label_embeddings
        h = self.final(h)
        return h

class ConditionalGANmodel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.automatic_optimization = False
        self.config = config
        self.generator = GENERATOR(config)
        self.discriminator = DISCRIMINATOR(config)
        self.k_d = config['k_d']
        self.k_g = config['k_g']
        self.st_res = config['generator:st_res']
        self.latent_dim = config['h_dim:generator'][0]
        self.num_classes = config['num_classes']
        self.sampling_period = config['sampling_period']
        self.grid = config['grid']

    def loss_function(self, x, labels, idx):
        # generator loss
        if idx == 0:
            z = torch.randn((x.shape[0], self.latent_dim, self.st_res, self.st_res), device=x.device)
            x_fake = self.generator(z, labels)
            ones = torch.ones((x.shape[0], 1), device=x.device)
            loss = F.binary_cross_entropy_with_logits(self.discriminator(x_fake, labels), ones)
            return loss
        # discriminator loss
        if idx == 1:
            z = torch.randn((x.shape[0], self.latent_dim, self.st_res, self.st_res), device=x.device)
            x_fake = self.generator(z, labels)
            
            # Real images should be classified as real (1)
            real_loss = F.binary_cross_entropy_with_logits(
                self.discriminator(x, labels), 
                torch.ones((x.shape[0], 1), device=x.device)
            )
            
            # Fake images should be classified as fake (0)
            fake_loss = F.binary_cross_entropy_with_logits(
                self.discriminator(x_fake.detach(), labels),  # detach to avoid training generator
                torch.zeros((x.shape[0], 1), device=x.device)
            )
            
            # Total discriminator loss is the average of real and fake losses
            loss = (real_loss + fake_loss) / 2
            return loss

        raise ValueError("Invalid idx value. Must be 0 for generator or 1 for discriminator.")

    def training_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device

        opt_g, opt_d = self.optimizers()
        # Discriminator Training
        for _ in range(self.k_d):
            opt_d.zero_grad()
            d_loss = self.loss_function(x, labels, idx=1)
            self.manual_backward(d_loss)
            opt_d.step()
            self.log('T:D_loss', d_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # Generator Training
        for _ in range(self.k_g):
            opt_g.zero_grad()
            g_loss = self.loss_function(x, labels, idx=0)
            self.manual_backward(g_loss)
            opt_g.step()
            self.log('T:G_loss', g_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return {"g_loss": g_loss, "d_loss": d_loss}

    def validation_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device

        with torch.no_grad():
            d_loss = self.loss_function(x, labels, idx=1)
            self.log('V:D_loss', d_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            # Generator Evaluation
            g_loss = self.loss_function(x, labels, idx=0)
            self.log('V:G_loss', g_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, labels = batch
        self.curr_device = x.device

        with torch.no_grad():
            # Discriminator Evaluation
            d_loss = self.loss_function(x, labels, idx=1)
            self.log('Te:D_loss', d_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

            # Generator Evaluation
            g_loss = self.loss_function(x, labels, idx=0)
            self.log('Te:G_loss', g_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.generator.parameters(),
            lr=self.config['learning_rate:g'],
            weight_decay=self.config['weight_decay']
        )
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config['learning_rate:d'],
            weight_decay=self.config['weight_decay']
        )

        optimizers = [opt_g, opt_d]

        try:
            if self.config['scheduler_gamma'] is not None:
                scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_g, gamma=self.config['scheduler_gamma'])
                scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_d, gamma=self.config['scheduler_gamma'])
                schedulers = [scheduler_g, scheduler_d]
                return optimizers, schedulers
        except KeyError:
            return optimizers
    
    def on_validation_epoch_end(self):
        if self.current_epoch % self.sampling_period == 0:
            self.sample_images()
    
    def sample(self, num_samples):
        
        z = torch.randn((num_samples, self.latent_dim, self.st_res, self.st_res), device=self.device)
        labels = torch.randint(0, self.num_classes, (num_samples,), device=self.device)

        with torch.no_grad():
            x_fake = self.generator(z, labels)
        return x_fake, labels

    def sample_images(self):

        try:
            samples, labels = self.sample(self.grid**2)
            
            samples_dir = os.path.join(self.logger.log_dir, "Samples")
            os.makedirs(samples_dir, exist_ok=True)

            vutils.save_image(samples.cpu().data,
                              os.path.join(samples_dir,
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=self.grid)
        except Warning:
            pass

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config, *args, **kwargs):
        model = cls(config, *args, **kwargs)
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        return model
