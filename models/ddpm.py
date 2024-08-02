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

def match_dims(x, y): return list(x.shape) + [1] * (len(y.shape)-len(x.shape))

class DownBlock(nn.Module):
    def __init__(self, in_features, out_features, act, with_conv=True):
        super().__init__()
        self.layers = nn.ModuleList()
        if with_conv:
            self.layers.append(nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=2, padding=1))
        else:
            self.layers.append(nn.AvgPool2d((2, 2)))

    def forward(self, x : Tensor, temb : Tensor) -> Tensor:
        h = x
        for module in self.layers:
            h = module(h)
        return h

class UpBlock(nn.Module):
    def __init__(self, in_features, out_features, act, with_conv=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        if with_conv:
            self.layers.append(nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=1, padding=1))

    def forward(self, x : Tensor, temb : Tensor) -> Tensor:
        h = x
        for module in self.layers:
            h = module(h)
        return h

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, act, time_features, conv_shortcut=False, dropout=0.):
        super().__init__()
        self.act = act
        self.linear_temb = Custumnn.Linear(time_features, out_features)
        self.block1 = nn.ModuleList()
        self.block1.append(nn.Sequential(
            nn.BatchNorm2d(in_features),
            self.act,
            nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=1, padding=1)
        ))
        self.block2 = nn.ModuleList()
        self.block2.append(nn.Sequential(
            nn.BatchNorm2d(out_features),
            self.act,
            nn.Dropout(dropout),
            nn.Conv2d(out_features, out_features, kernel_size=(3, 3), stride=1, padding=1)
        ))
        
        self.layers = None
        if in_features!= out_features:
            self.layers = nn.ModuleList()
            if conv_shortcut:
                self.layers.append(nn.Conv2d(out_features, out_features, kernel_size=(3, 3), stride=1, padding=1)) # x to x
            else:
                self.layers.append(Custumnn.Linear(in_features, out_features)) # x to x
    def forward(self, x : Tensor, temb=None) -> Tensor:
        if temb==None:
            temb = torch.randint(0, self.t_max, (x.shape[0], 64), device=x.device, dtype=x.dtype)
        h = x
        for module in self.block1:
            h = module(h)
        
        temb = self.linear_temb(self.act(temb))
        h+= temb

        for module in self.block2:
            h = module(h)

        if self.layers != None:
            for module in self.layers:
                x = module(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.linear1 = Custumnn.Linear(features, features)
        self.linear2 = Custumnn.Linear(features, features)
        self.norm = nn.BatchNorm2d(features)

    def forward(self, x : Tensor, temb : Tensor) -> Tensor:
        
        h = self.norm(x)
        q = self.linear1(h).view(h.shape[0], h.shape[1], h.shape[2] * h.shape[3])
        k = self.linear1(h).view(h.shape[0], h.shape[1], h.shape[2] * h.shape[3])
        v = self.linear1(h).view(h.shape[0], h.shape[1], h.shape[2] * h.shape[3])
        w = torch.matmul(q, k.transpose(-2, -1)) * (int(k.shape[1]) ** (-0.5))
        w = F.softmax(w, dim=-1)
        h = torch.matmul(w, v).view(h.shape)
        h = self.linear2(h)
        return x + h

def timeembed(t, embedding_dim):
    frequencies = torch.exp(
        torch.linspace(
            torch.math.log(1.0),
            torch.math.log(1000.0),
            embedding_dim // 2,
            device=t.device
        ).reshape(1, embedding_dim // 2, 1, 1)
    )
    angular_speeds = 2.0 * math.pi * frequencies
    embeddings = torch.concat([torch.sin(angular_speeds * t), torch.cos(angular_speeds * t)], dim=1)
    return embeddings

class DDPMModel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.h_dims = config['h_dims'] # [3, 16, 32, 64, 128, 256]
        self.patch_size = config['patch_size']
        self.act = nn.LeakyReLU()
        self.attn_resolutions = config['attn_res']
        self.down_index = config['down_index']
        self.with_conv = config['with_conv']
        self.dropout = config['dropout']
        self.num_channels = len(self.h_dims)
        self.latent_dim = config['latent_dim']
        self.sampling_period = config['sampling_period']
        self.diffusion_steps = config['diffusion_steps']
        self.grid = config['grid']
        self.ch = self.h_dims[1]
        
        self.linear_1 = Custumnn.Linear(self.h_dims[1], self.h_dims[1] * 4)
        self.linear_2 = Custumnn.Linear(self.h_dims[1] * 4, self.h_dims[1] * 4)
        self.time_features = self.h_dims[1] * 4
        
        # Downsample
        self.Downsampling = nn.ModuleList()
        
        f_res = self.patch_size # [224, 112, 56, 28, 14, 7]
        for index in range(0, self.num_channels-1):
            in_channels = self.h_dims[index]
            out_channels = self.h_dims[index+1]
            self.Downsampling.append(
                ResNetBlock(in_features=in_channels, out_features=out_channels, act=self.act, time_features=self.time_features)
            )
            if f_res in self.attn_resolutions:
                self.Downsampling.append(AttnBlock(features=out_channels))
            # Downsample
            if self.down_index[index+1]==1:
                self.Downsampling.append(DownBlock(in_features=out_channels, out_features=out_channels, act=self.act, with_conv=self.with_conv))
                f_res /= 2
        
        # Middle
        self.Middle = nn.ModuleList()
        self.Middle.append(ResNetBlock(out_channels, out_channels, self.act, self.time_features, dropout=self.dropout))
        self.Middle.append(AttnBlock(out_channels))
        self.Middle.append(ResNetBlock(out_channels, out_channels, self.act, self.time_features, dropout=self.dropout))

        # Upsampling
        self.reversed_h_dims = self.h_dims[::-1]
        self.reversed_down_index = self.down_index[::-1]
        self.Upsampling = nn.ModuleList()
        for index in range(0, self.num_channels-1):
            in_channels = self.reversed_h_dims[index]
            out_channels = self.reversed_h_dims[index+1]
            # Upsample
            if self.reversed_down_index[index]==1:
                self.Upsampling.append(UpBlock(in_features=in_channels, out_features=in_channels, act=self.act, with_conv=self.with_conv))
                f_res *= 2
            self.Upsampling.append(
                ResNetBlock(in_features=in_channels, out_features=out_channels, act=self.act, time_features=self.time_features)
            )
            if f_res in self.attn_resolutions:
                self.Upsampling.append(AttnBlock(features=out_channels))
        # End
        self.End = nn.ModuleList()
        self.End.append(
            nn.Sequential(
                Custumnn.Linear(out_channels, out_channels),
            ))
        self.train_losses = []
        self.val_losses = []
    def forward(self, x : Tensor, t= None) -> Tensor:
        if t == None:
            print("Test mode. It can't be trained.")
            t = torch.rand((x.shape[0],1, 1, 1), device=x.device) * 0
        h = x
        # Timestep embedding
        temb = timeembed(t, self.ch)
        temb = self.linear_1(temb)
        temb = self.linear_2(self.act(temb))

        skip = []
        # Downsampling
        for module in self.Downsampling:
            h = module(h, temb)
            skip.append(h)
        # Middle
        for module in self.Middle:
            h = module(h, temb)
        # Upsampling
        skip.reverse()
        for index, module in enumerate(self.Upsampling):
            if index < len(skip):
                h += skip[index]
            h = module(h, temb)
        # for module in self.Upsampling:
        #     h = module(h, temb)
        # End
        for module in self.End:
            h = module(h)
        return h
    # Cosine scheduler
    def diffusion_schedule(self, diffusion_times: int) -> List[Tensor]:
        min_signal_rate = 0.02
        max_signal_rate = 0.95
        min_signal_rate = torch.tensor(min_signal_rate, device=self.curr_device)
        max_signal_rate = torch.tensor(max_signal_rate, device=self.curr_device)

        start_angle = torch.acos(max_signal_rate)
        end_angle = torch.acos(min_signal_rate)
        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)

        return noise_rates, signal_rates

    def denoise(self, x_t : Tensor, noise_rates, signal_rates) -> Tensor:
        pred_noises = self(x_t, noise_rates**2)
        pred_images = (x_t - noise_rates * pred_noises) / signal_rates # mu = (x_t - beta/root(1-alpha_bar)eps) / root(alpha)
        return pred_noises, pred_images

    def p_loss(self, images):
        noises = torch.randn_like(images, dtype=images.dtype, device=images.device)
        diffusion_times = torch.rand((images.shape[0], 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times) # \sqrt{alpha_bar}, \sqrt{alpha_bar}
        noisy_images = signal_rates * images + noise_rates * noises
        pred_noises, pred_images = self.denoise(noisy_images, noise_rates, signal_rates)
        loss = F.l1_loss(pred_noises, noises)
        return loss

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        for step in range(diffusion_steps):
            diffusion_times = torch.ones((num_images, 1, 1, 1), device=self.curr_device) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                current_images, noise_rates, signal_rates
            )
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            current_images = (
                next_signal_rates * pred_images + next_noise_rates * pred_noises # x_{t-1} = mu_\theta + sigma * noise
            )
        return pred_images

    def generate(self, num_images, diffusion_steps, initial_noise=None):
        if initial_noise is None:
            initial_noise = torch.randn((num_images, 3, self.patch_size, self.patch_size), device=self.curr_device)
        generated_images = self.reverse_diffusion(
            initial_noise, diffusion_steps
        )
        return generated_images
    # ============================================================
    # ============================================================
    # ============================================================

    def save_loss(self, phase, loss):
        save_path = os.path.join(self.logger.log_dir, f'{phase}_losses.npy')
        if self.trainer.is_global_zero: # Preventing duplication for multi GPU environment
            try:
                losses = np.load(save_path)
                losses = np.append(losses, loss)
            except FileNotFoundError:
                losses = np.array([loss])
            np.save(save_path, losses)

    def training_step(self, batch, batch_idx):
        x_0, y = batch
        self.curr_device = x_0.device
        loss = self.p_loss(x_0)
        self.log('TL', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        self.train_losses.append(loss.item())
        return loss

    def on_train_epoch_end(self):
        avg_loss = np.mean(self.train_losses)
        self.train_losses = []
        self.save_loss('train', avg_loss)

    def validation_step(self, batch, batch_idx):
        x_0, y = batch
        self.curr_device = x_0.device
        loss = self.p_loss(x_0)
        self.log('VL', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.val_losses.append(loss.item())

    def on_validation_end(self) -> None:
        if self.current_epoch % self.sampling_period == 0:
            self.sample_images()

    def test_step(self, batch, batch_idx):
        x_0, y = batch
        self.curr_device = x_0.device
        loss = self.p_loss(x_0)
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

    def sample_images(self):

        try:
            samples = self.generate(self.grid**2, self.diffusion_steps)
            
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