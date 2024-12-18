import os
from typing import *
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torchvision.utils as vutils
from .cnn import CNNModel
from safetensors.torch import save_file, load_file
import numpy as np
from scipy import linalg
import scipy

def calculate_activation_statistics(features):
    """Calculate mean and covariance statistics of features using PyTorch."""
    mu = torch.mean(features, dim=0)
    sigma = torch.cov(features.T)  # features.T because torch.cov expects shape (features, observations)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Calculate Frechet Distance between two multivariate Gaussians using PyTorch."""
    """
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))."""
    mu1 = mu1.view(-1)
    mu2 = mu2.view(-1)
    
    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2
    
    # Product might be almost singular
    covmean = torch.from_numpy(scipy.linalg.sqrtm(sigma1.cpu().numpy() @ sigma2.cpu().numpy())).to(mu1.device)
    
    if not torch.isfinite(covmean).all():
        print(f'fid calculation produces singular product; adding {eps} to diagonal of cov estimates')
        offset = torch.eye(sigma1.shape[0], device=mu1.device) * eps
        covmean = torch.from_numpy(
            scipy.linalg.sqrtm((sigma1.cpu().numpy() + offset.cpu().numpy()) @ (sigma2.cpu().numpy() + offset.cpu().numpy()))
        ).to(mu1.device)

    # Numerical error might give slight imaginary component
    if torch.is_complex(covmean):
        if not torch.allclose(torch.diagonal(covmean).imag, torch.zeros_like(torch.diagonal(covmean).imag), atol=1e-3):
            m = torch.max(torch.abs(covmean.imag))
            raise ValueError(f"Imaginary component {m}")
        covmean = covmean.real

    tr_covmean = torch.trace(covmean)
    return diff @ diff + torch.trace(sigma1) + torch.trace(sigma2) - 2 * tr_covmean

class BVAEModel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.h_dims = config['h_dims'] # [3, 16, 32, 64, 128, 256]
        self.patch_size = config['patch_size']
        self.latent_dim = config['latent_dim']
        self.kld_weight = config['kld_weight']
        self.sampling_period = config['sampling_period']
        self.num_z = config['num_z']
        self.prior = config['prior']
        self.cnn_config = config['cnn_config']
        num_channels = len(self.h_dims)

        cnn_model = CNNModel(self.cnn_config)
        checkpoint_path = 'logs/mnist_cnn/version_0/checkpoints/last.safetensors'
        cnn_model = cnn_model.__class__.load_from_checkpoint(checkpoint_path, config=self.cnn_config)
        self.cnn_model = cnn_model

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
                nn.SiLU(),
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
                nn.SiLU(),
            ))
            f_res *= 2
        # Final
        self.Upsampling.append(nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1))
        
        self.real_features = []
        self.fake_features = []
    def forward(self, x, test = None):
        h = x

        # Encoder
        h = self.encode(h)

        # Sampling
        flatten = h.view([h.shape[0], -1])
        mu = self.FC_mu(flatten)
        logvar = self.FC_logvar(flatten)
        if self.prior=='exponential':
            mu = torch.sqrt(nn.functional.relu(mu)+0.01)
        z = self.reparam(mu, logvar, test)

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

    def reparam(self, mu, logvar, test = None):
        # Input : [B, F] Output : [B x L, F}
        if self.prior == 'normal':
            if test:
                return mu
            else:
                self.batch_size, F = mu.shape
                std = torch.exp(logvar.unsqueeze(1).repeat(1, self.num_z, 1) * 0.5)
                eps = torch.randn_like(std)
                result = mu.unsqueeze(1) + eps * std
                return result.view(self.batch_size * self.num_z, F)
        
        elif self.prior == 'exponential':
            if test:
                return mu
            else:
                def exponential_like(x, rate=1.0):
                    m = torch.distributions.Exponential(rate=torch.full_like(x, rate))
                    return m.rsample()
                self.batch_size, F = mu.shape
                eps = exponential_like(mu.unsqueeze(1).repeat(1, self.num_z, 1))
                result = eps / mu.unsqueeze(1)
                return result.view(self.batch_size * self.num_z, F)

                # rate=(1.0/mu).unsqueeze(1).repeat(1, self.num_z, 1)
                # exponential = torch.distributions.Exponential(rate=rate)  # rate = 1/μ
                # samples = exponential.rsample()  # [L, B, F]
                # return samples.permute(1, 0, 2).reshape(self.batch_size * self.num_z, F)  # [B*L, F]
        
        elif self.prior == 'laplace':
            if test:
                return mu
            else:
                def laplace_like(x, rate=1.0):
                    m = torch.distributions.Laplace(loc=torch.zeros_like(x), scale=torch.full_like(x, rate))
                    return m.rsample()
                self.batch_size, F = mu.shape
                eps = laplace_like(mu.unsqueeze(1).repeat(1, self.num_z, 1))
                scale = torch.exp(logvar).sqrt().unsqueeze(1).repeat(1, self.num_z, 1)  # b parameter
                result = mu.unsqueeze(1) + eps * scale
                return result.view(self.batch_size * self.num_z, F)

                # scale = torch.exp(logvar).sqrt().unsqueeze(1).repeat(1, self.num_z, 1)  # b parameter
                # laplace = torch.distributions.Laplace(loc=mu.unsqueeze(1), scale=scale)
                # samples = laplace.rsample()  # rsample()은 reparameterized sampling을 수행
                # return samples.view(self.batch_size * self.num_z, F)

        
        else:
            raise ValueError(f"Unknown prior distribution: {self.prior}")

    def training_step(self, batch, batch_idx):
        x, _ = batch
        _, C, H, W = x.shape
        self.curr_device = x.device
        x_rec, mu, logvar = self(x)
        x_rec = x_rec.view(self.batch_size, -1, C, H, W)
        rec_loss = F.mse_loss(x_rec, x.unsqueeze(1).repeat(1, self.num_z, 1, 1, 1))
        if self.prior == 'normal':
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        elif self.prior == 'exponential':
            kld_loss = 0.5 * torch.mean(-1 * torch.sum(1 + torch.log(mu) - mu, dim = 1), dim = 0)
        elif self.prior == 'laplace':
            kld_loss = 0.5 * torch.mean(-1 * torch.sum(1 + logvar*0.5 - torch.abs(mu) - (logvar*0.5).exp(), dim = 1), dim = 0)
        loss = rec_loss + self.kld_weight * kld_loss
        self.log('TL', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        _, C, H, W = x.shape
        self.curr_device = x.device
        x_rec, mu, logvar = self(x)
        x_rec = x_rec.view(self.batch_size, -1, C, H, W)
        rec_loss = F.mse_loss(x_rec, x.unsqueeze(1).repeat(1, self.num_z, 1, 1, 1))
        if self.prior == 'normal':
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        elif self.prior == 'exponential':
            kld_loss = 0.5 * torch.mean(-1 * torch.sum(1 + torch.log(mu) - mu, dim = 1), dim = 0)
        elif self.prior == 'laplace':
            kld_loss = 0.5 * torch.mean(-1 * torch.sum(1 + logvar*0.5 - torch.abs(mu) - (logvar*0.5).exp(), dim = 1), dim = 0)
        loss = rec_loss + self.kld_weight * kld_loss
        self.log('VL', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        _, C, H, W = x.shape
        self.curr_device = x.device
        x_rec, mu, logvar = self(x)
        x_rec = x_rec.view(self.batch_size, -1, C, H, W)
        rec_loss = F.mse_loss(x_rec, x.unsqueeze(1).repeat(1, self.num_z, 1, 1, 1))
        if self.prior == 'normal':
            kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        elif self.prior == 'exponential':
            kld_loss = torch.mean(-1 * torch.sum(1 + torch.log(mu) - mu, dim = 1), dim = 0)
        elif self.prior == 'laplace':
            kld_loss = torch.mean(-1 * torch.sum(1 + logvar*0.5 - torch.abs(mu) - (logvar*0.5).exp(), dim = 1), dim = 0)
        loss = rec_loss + self.kld_weight * kld_loss

        # 3. Feature 수집
        with torch.no_grad():
            # 실제 이미지의 feature 추출
            real_features = self.cnn_model.extract_features(x)
            self.real_features.append(real_features.cpu())
            
            # prior에서 샘플링한 z로부터 이미지 생성
            fake_images = self.sample(x.size(0))
            fake_features = self.cnn_model.extract_features(fake_images)
            self.fake_features.append(fake_features.cpu())

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

    def on_test_epoch_end(self):
        # 모든 feature를 하나로 합치기
        real_features = torch.cat(self.real_features, dim=0)
        fake_features = torch.cat(self.fake_features, dim=0)
        
        # feature statistics 계산
        mu_real, sigma_real = calculate_activation_statistics(real_features)
        mu_fake, sigma_fake = calculate_activation_statistics(fake_features)
        
        # FID score 계산
        fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        
        # Score 로깅
        self.log('FID', fid_score, on_epoch=True, sync_dist=True)
        
        # 리스트 초기화
        self.real_features = []
        self.fake_features = []

    def sample(self, num_samples):
        if self.prior == 'normal':
            z = torch.randn(num_samples, self.latent_dim, device=self.curr_device)
        elif self.prior == 'exponential':
            exponential = torch.distributions.Exponential(rate=torch.ones(num_samples, self.latent_dim, device=self.curr_device))
            z = exponential.rsample()
        elif self.prior == 'laplace':
            laplace = torch.distributions.Laplace(
                loc=torch.zeros(num_samples, self.latent_dim, device=self.curr_device),
                scale=torch.ones(num_samples, self.latent_dim, device=self.curr_device)
            )
            z = laplace.rsample()

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

        recons = self(test_input, test=True)[0]
        recons_dir = os.path.join(self.logger.log_dir, "Reconstructions")
        os.makedirs(recons_dir, exist_ok=True)

        vutils.save_image(recons.data,
                          os.path.join(recons_dir,
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=4)
        
        original_dir = os.path.join(self.logger.log_dir, "Original")
        os.makedirs(original_dir, exist_ok=True)
        vutils.save_image(test_input,
                          os.path.join(original_dir,
                                       f"original_{self.logger.name}_Epoch_{self.current_epoch}.png"),
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
