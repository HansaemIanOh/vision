import os
import math
from typing import *
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision.utils as vutils
from . import Custumnn

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
        h+= temb.view(match_dims(temb, h))

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

def timeembed(timesteps: Tensor, embedding_dim: int) -> Tensor:

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))

    return emb

class FLOWERDDPMModel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        h_dims = config['h_dims'] # [3, 16, 32, 64, 128, 256]
        patch_size = config['patch_size']
        self.act = nn.LeakyReLU()
        self.attn_resolutions = config['attn_res']
        self.with_conv = config['with_conv']
        self.dropout = config['dropout']
        num_channels = len(h_dims)
        self.t_min = 0
        self.t_max = config['t_max']
        self.latent_dim = config['latent_dim']
        self.sampling_period = config['sampling_period']
        self.warmup_step = config['warmup_step']
        self.ch = h_dims[1]
        
        self.linear_1 = Custumnn.Linear(h_dims[1], h_dims[1] * 4)
        self.linear_2 = Custumnn.Linear(h_dims[1] * 4, h_dims[1] * 4)
        self.time_features = h_dims[1] * 4
        
        # Downsample
        self.Downsampling = nn.ModuleList()
        
        f_res = patch_size # [224, 112, 56, 28, 14, 7]
        for index in range(0, num_channels-1):
            in_channels = h_dims[index]
            out_channels = h_dims[index+1]
            self.Downsampling.append(
                ResNetBlock(in_features=in_channels, out_features=out_channels, act=self.act, time_features=self.time_features)
            )
            if f_res in self.attn_resolutions:
                self.Downsampling.append(AttnBlock(features=out_channels))
            # Downsample
            self.Downsampling.append(DownBlock(in_features=out_channels, out_features=out_channels, act=self.act, with_conv=self.with_conv))
            f_res /= 2
        
        # Middle
        self.Middle = nn.ModuleList()
        self.Middle.append(ResNetBlock(out_channels, out_channels, self.act, self.time_features, dropout=self.dropout))
        self.Middle.append(AttnBlock(out_channels))
        self.Middle.append(ResNetBlock(out_channels, out_channels, self.act, self.time_features, dropout=self.dropout))

        # Upsampling
        h_dims.reverse()
        self.Upsampling = nn.ModuleList()
        for index in range(0, num_channels-1):
            in_channels = h_dims[index]
            out_channels = h_dims[index+1]
            # Upsample
            self.Upsampling.append(UpBlock(in_features=in_channels, out_features=in_channels, act=self.act, with_conv=self.with_conv))
            self.Upsampling.append(
                ResNetBlock(in_features=in_channels, out_features=out_channels, act=self.act, time_features=self.time_features)
            )
            if f_res in self.attn_resolutions:
                self.Upsampling.append(AttnBlock(features=out_channels))
            f_res *= 2
                # End
        self.End = nn.ModuleList()
        self.End.append(nn.Sequential(
            nn.GroupNorm(num_groups=out_channels, num_channels=out_channels),
            self.act,
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        ))
    
    def forward(self, x : Tensor, t= None) -> Tensor:
        if t == None:
            print("Test mode. It can't be trained.")
            t = t+1
            t = torch.randint(low=self.t_min, high=self.t_max, size=(x.shape[0],), device=x.device)
        h = x
        # Timestep embedding
        temb = timeembed(t, self.ch)

        temb = self.linear_1(temb)
        temb = self.linear_2(self.act(temb))

        # Downsampling
        for module in self.Downsampling:
            # print(h.shape)
            h = module(h, temb)
        # Middle
        for module in self.Middle:
            # print(h.shape)
            h = module(h, temb)
        # Upsampling
        for module in self.Upsampling:
            # print(h.shape)
            h = module(h, temb)
        # End
        for module in self.End:
            # print(h.shape)
            h = module(h)

        return h

    def generate_noise(self, x_0: Tensor, seeds=None) -> Tensor:
        B, C, H, W = x_0.shape
        device = x_0.device
        if seeds==None:
            seeds = torch.sum(x_0.view(B, -1), dim=1).long()
        noises = []
        for seed in seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed.item()))
            noise = torch.randn(C, H, W, generator=generator, device=device)
            noises.append(noise)

        return torch.stack(noises)
    def training_step(self, batch, batch_idx):
        x_0, y = batch
        self.curr_device = x_0.device

        if self.current_epoch<self.warmup_step:
            seeds = torch.ones((x_0.shape[0],), device=self.curr_device).long()*42
        else:
            seeds = None
        noise = self.generate_noise(x_0, seeds)
        t = torch.randint(self.t_min, self.t_max, (x_0.shape[0],), device=x_0.device)
        loss = self.p_losses(x_0, t, model=self, eps=noise)
        self.log('TL', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x_0, y = batch
        self.curr_device = x_0.device

        if self.current_epoch<self.warmup_step:
            seeds = torch.ones((x_0.shape[0],), device=self.curr_device).long()*42
        else:
            seeds = None
        noise = self.generate_noise(x_0, seeds)
        t = torch.randint(self.t_min, self.t_max, (x_0.shape[0],), device=x_0.device)
        loss = self.p_losses(x_0, t, model=self, eps=noise)
        self.log('Val Loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x_0, y = batch
        self.curr_device = x_0.device
        t = torch.randint(self.t_min, self.t_max, (x_0.shape[0],), device=x_0.device)
        loss = self.p_losses(x_0, t, model=self)
        self.log('Test Loss', loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])

    def on_validation_end(self) -> None:
        if self.current_epoch % self.sampling_period == 0:
            self.sample_images()
    # ============================================================
    # ============================================================
    # ============================================================

    def sample_images(self):
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        
        seeds = torch.ones((test_input.shape[0],), device=self.curr_device)
        noise = self.generate_noise(test_input, seeds)
        recons = self.sample(16, current_device=self.curr_device, noise=noise)
        recons_dir = os.path.join(self.logger.log_dir, "Reconstructions")
        os.makedirs(recons_dir, exist_ok=True)

        vutils.save_image(recons.data,
                          os.path.join(recons_dir,
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=4)
        
        try:
            samples = self.sample(16, current_device=self.curr_device)
            
            samples_dir = os.path.join(self.logger.log_dir, "Samples")
            os.makedirs(samples_dir, exist_ok=True)

            vutils.save_image(samples.cpu().data,
                              os.path.join(samples_dir,
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=4)
        except Warning:
            pass
    
    def beta_func(self, t): # return beta
        beta_min = 0.0001
        beta_max = 0.02
        beta_t = beta_min + (beta_max - beta_min) * (1 - t / self.t_max)
        return beta_t

    def alpha_func(self, t): # return alpha
        return 1 - self.beta_func(t)

    def alpha_cumprod_func(self, t):
        alphas = self.alpha_func(torch.arange(int(t.max()) + 1, dtype=torch.float32, device=t.device))
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return alphas_cumprod[t.long()]

    def q_sample(self, x_start : Tensor, t, eps): # q(x_t | x_0)
        alpha_bar = self.alpha_cumprod_func(t)
        alpha_bar = alpha_bar.view(match_dims(alpha_bar, x_start))
        c_1 = torch.sqrt(alpha_bar)
        c_2 = torch.sqrt(1-alpha_bar)
        x_noisy = c_1 * x_start + c_2 * eps
        return x_noisy

    def q_reverse(self, x_t : Tensor, t, eps_theta): # q(x_0 | x_t)
        alpha_bar = self.alpha_cumprod_func(t)
        alpha_bar = alpha_bar.view(match_dims(alpha_bar, x_t))
        x_start = (x_t - torch.sqrt(1 - alpha_bar) * eps_theta) / torch.sqrt(alpha_bar)
        return x_start

    def q_posterior(self, x_start, x_t : Tensor, t, sigma_cal=False):
        """
        Compute the mean and variance of the diffusion posterior q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        beta = self.beta_func(t)
        alpha = self.alpha_func(t)
        alpha_bar_t = self.alpha_cumprod_func(t)
        alpha_bar_t_1 = self.alpha_cumprod_func(t-1)

        assert torch.all(t > 0), "Time in p_sample is zero"
        
        c_0 = (torch.sqrt(alpha_bar_t_1) * beta) / ((1 - alpha_bar_t))
        c_t = (torch.sqrt(alpha_bar_t) * (1 - alpha_bar_t_1)) / ((1 - alpha_bar_t))
        
        c_0 = c_0.view(match_dims(c_0, x_start))
        c_t = c_t.view(match_dims(c_t, x_t))

        mean = c_0 * x_start + c_t * x_t
        if sigma_cal:
            sigma_squared = (1 - self.alpha_cumprod_func(t-1)) / (1 - self.alpha_cumprod_func(t)) * beta
        else:
            sigma_squared = beta
        sigma = torch.sqrt(sigma_squared)

        return mean, sigma
    
    def p_losses(self, x_start : Tensor, t, eps=None, model=None):
        # x_start, t, model 을 받아서 x_t, t를 만들고 그걸 모델에 넣어서 loss를 만듦.
        assert model!=None, "Model is none"
        if eps is None:
            eps = torch.randn_like(x_start, dtype=x_start.dtype, device=x_start.device)
        
        x_t = self.q_sample(x_start=x_start, t=t, eps=eps) # q(x_t | x_0) \eps_\theta(c_1 x_0 + c_2 \eps, t)
        eps_theta = model(x_t, t)

        losses = F.mse_loss(eps_theta, eps)

        return losses # scalar

    def p_sample(self, x_t : Tensor, t, eps=None, model=None, sigma_cal=False):

        assert model!=None, "Model is none"
        if eps is None:
            eps = torch.randn_like(x_t, dtype=x_t.dtype, device=x_t.device)

        eps_theta = model(x_t, t)
        
        x_start = self.q_reverse(x_t, t, eps_theta)
        
        mean, sigma = self.q_posterior(x_t=x_t, t=t, x_start=x_start, sigma_cal=sigma_cal)
        sigma = sigma.view(match_dims(sigma, x_t))
        x_t_1 = mean + sigma * eps_theta # x_t_1 is x_{t-1}
        return x_t_1
    
    def sample(self,
               num_samples:int,
               current_device: int, 
               noise=None,
               **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        x_T = torch.randn([num_samples] + self.latent_dim, device=current_device)
        if noise == None:
            noise = x_T
        # ==============================
        x_T = noise
        # ==============================
        x_t = x_T
        T = self.t_max
        t_tensor = torch.ones((x_T.shape[0],), device=current_device) * T

        t=self.t_max
        while t>=1:
            x_t_1 = self.p_sample(x_t, t_tensor, model=self, eps=noise)
            t_tensor -= 1
            t-=1
        x_1 = x_t_1
        # x_0 = self.p_final(x_1)
        x_0 = x_1
        return x_0
    

