import os
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

class VAEModel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.h_dims = config['h_dims'] # [3, 16, 32, 64, 128, 256]
        self.patch_size = config['patch_size']
        self.latent_dim = config['latent_dim']
        self.kld_weight = config['kld_weight']
        self.sampling_period = config['sampling_period']
        num_channels = len(self.h_dims)
        # Downsample
        self.Downsampling = nn.ModuleList()
        
        f_res = self.patch_size # [224, 112, 56, 28, 14, 7]
        for index in range(0, num_channels-1):
            in_channels = self.h_dims[index]
            out_channels = self.h_dims[index+1]

            self.Downsampling.append(
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))
            f_res /= 2

        self.latent_res = int(f_res)
        self.latent_ch = int(out_channels)
        self.FC_mu = nn.Linear(self.latent_ch*self.latent_res**2, self.latent_dim)
        self.FC_logvar = nn.Linear(self.latent_ch*self.latent_res**2, self.latent_dim)
        self.FC_restore = nn.Linear(self.latent_dim, self.latent_ch*self.latent_res**2)
        # Upsample
        self.Upsampling = nn.ModuleList()

        reversed_h_dims = self.h_dims[::-1]
        for index in range(0, num_channels-1):
            in_channels = reversed_h_dims[index]
            out_channels = reversed_h_dims[index+1]

            self.Upsampling.append(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ))
            f_res *= 2
        # Final
        self.Upsampling.append(
            nn.Sequential(
                Custumnn.Linear(out_channels, out_channels),
            ))

    def forward(self, x):
        h = x

        # Encoder
        h = self.encode(h)

        # Sampling
        flatten = h.view([h.shape[0], -1])
        mu = self.FC_mu(flatten)
        logvar = self.FC_logvar(flatten)
        z = self.reparam(mu, logvar)

        # Retoration
        z = self.FC_restore(z)
        h = z.view([-1, self.latent_ch, self.latent_res, self.latent_res])
        # Decoder
        h = self.decode(h)
        return [h, mu, logvar]

    def encode(self, x):
        h = x
        # Down
        for module in self.Downsampling:
            h = module(h)
        return h

    def decode(self, x):
        h = x
        # Up
        for module in self.Upsampling:
            h = module(h)
        return h

    def reparam(self, mu, logvar):
        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        self.curr_device = x.device
        x_rec, mu, logvar = self(x)
        rec_loss = F.mse_loss(x_rec, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        loss = rec_loss + self.kld_weight * kld_loss
        self.log('TL', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        self.curr_device = x.device
        x_rec, mu, logvar = self(x)
        rec_loss = F.mse_loss(x_rec, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        loss = rec_loss + self.kld_weight * kld_loss
        self.log('VL', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_rec, mu, logvar = self(x)
        rec_loss = F.mse_loss(x_rec, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        loss = rec_loss + self.kld_weight * kld_loss
        self.log('TeL', loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = \
        torch.optim.AdamW(
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
    
    def on_validation_epoch_end(self):
        if self.current_epoch % self.sampling_period == 0:
            self.sample_images()
    
    def sample(self, num_samples):
        
        z = torch.randn(num_samples, self.latent_dim, device=self.curr_device)

        # Retoration
        z = self.FC_restore(z)
        h = z.view([-1, self.latent_ch, self.latent_res, self.latent_res])
        
        # Decoder
        h = self.decode(h)
        return h
    def sample_images(self):
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        recons = self(test_input)[0]
        recons_dir = os.path.join(self.logger.log_dir, "Reconstructions")
        os.makedirs(recons_dir, exist_ok=True)

        vutils.save_image(recons.data,
                          os.path.join(recons_dir,
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=4)

        try:
            samples = self.sample(16)
            
            samples_dir = os.path.join(self.logger.log_dir, "Samples")
            os.makedirs(samples_dir, exist_ok=True)

            vutils.save_image(samples.cpu().data,
                              os.path.join(samples_dir,
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=4)
        except Warning:
            pass

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, config, *args, **kwargs):
        model = cls(config, *args, **kwargs)
        state_dict = load_file(checkpoint_path)
        model.load_state_dict(state_dict, strict=False)
        return model