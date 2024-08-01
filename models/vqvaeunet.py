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
def match_dims(x, y): return list(x.shape) + [1] * (len(y.shape)-len(x.shape))

class DownBlock(nn.Module):
    def __init__(self, in_features, out_features, act, with_conv=True):
        super().__init__()
        self.layers = nn.ModuleList()
        if with_conv:
            self.layers.append(nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=2, padding=1))
        else:
            self.layers.append(nn.AvgPool2d((2, 2)))

    def forward(self, x : Tensor) -> Tensor:
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

    def forward(self, x : Tensor) -> Tensor:
        h = x
        for module in self.layers:
            h = module(h)
        return h

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, act, conv_shortcut=False, dropout=0.):
        super().__init__()
        self.act = act
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
    def forward(self, x : Tensor) -> Tensor:
        h = x
        for module in self.block1:
            h = module(h)
        
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

    def forward(self, x : Tensor) -> Tensor:
        
        h = self.norm(x)
        q = self.linear1(h).view(h.shape[0], h.shape[1], h.shape[2] * h.shape[3])
        k = self.linear1(h).view(h.shape[0], h.shape[1], h.shape[2] * h.shape[3])
        v = self.linear1(h).view(h.shape[0], h.shape[1], h.shape[2] * h.shape[3])
        w = torch.matmul(q, k.transpose(-2, -1)) * (int(k.shape[1]) ** (-0.5))
        w = F.softmax(w, dim=-1)
        h = torch.matmul(w, v).view(h.shape)
        h = self.linear2(h)
        return x + h



class VQBlock(nn.Module):

    def __init__(self, num_embed: int, embed_dim: int, beta: float = 0.25):
        super().__init__()
        self.K = num_embed
        self.D = embed_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.K, self.D)
        self.embedding.weight.data.uniform_(-1 / self.K, 1 / self.K)

    def forward(self, latents: Tensor) -> Tensor:
        latents = latents.permute(0, 2, 3, 1).contiguous()  # [B x D x H x W] -> [B x H x W x D]
        latents_shape = latents.shape
        flat_latents = latents.view(-1, self.D)  # [BHW x D]

        # Compute L2 distance between latents and embedding weights
        dist = torch.sum(flat_latents ** 2, dim=1, keepdim=True) + \
               torch.sum(self.embedding.weight ** 2, dim=1) - \
               2 * torch.matmul(flat_latents, self.embedding.weight.t())  # [BHW x K]

        # Get the encoding that has the min distance
        encoding_inds = torch.argmin(dist, dim=1).unsqueeze(1)  # [BHW, 1]

        # Convert to one-hot encodings
        device = latents.device
        encoding_one_hot = torch.zeros(encoding_inds.size(0), self.K, device=device)
        encoding_one_hot.scatter_(1, encoding_inds, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), latents)
        embedding_loss = F.mse_loss(quantized_latents, latents.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = latents + (quantized_latents - latents).detach()

        return quantized_latents.permute(0, 3, 1, 2).contiguous(), vq_loss  # [B x D x H x W]


class VQVAEUNETModel(pl.LightningModule):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.h_dims = config['h_dims'] # [3, 16, 32, 64, 128, 256]
        self.patch_size = config['patch_size']
        self.num_embed = config['num_embed']
        self.embed_dim = config['embed_dim']
        self.sampling_period = config['sampling_period']
        self.attn_resolutions = config['attn_res']
        self.down_index = config['down_index']
        self.act = nn.LeakyReLU()
        self.with_conv = True
        num_channels = len(self.h_dims)
        self.vq_layer = VQBlock(self.num_embed, self.embed_dim)
        # Downsample
        self.Downsampling = nn.ModuleList()
        
        f_res = self.patch_size # [224, 112, 56, 28, 14, 7]
        for index in range(0, num_channels-1):
            in_channels = self.h_dims[index]
            out_channels = self.h_dims[index+1]
            
            self.Downsampling.append(
                ResNetBlock(in_features=in_channels, out_features=out_channels, act=self.act)
            )
            if f_res in self.attn_resolutions:
                self.Downsampling.append(AttnBlock(features=out_channels))
            # Downsample
            if self.down_index[index+1]==1:
                self.Downsampling.append(DownBlock(in_features=out_channels, out_features=out_channels, act=self.act, with_conv=self.with_conv))
                f_res /= 2
        
        # Upsampling
        self.h_dims.reverse()
        self.down_index.reverse()
        self.Upsampling = nn.ModuleList()
        for index in range(0, num_channels-1):
            in_channels = self.h_dims[index]
            out_channels = self.h_dims[index+1]
            # Upsample
            if self.down_index[index]==1:
                self.Upsampling.append(UpBlock(in_features=in_channels, out_features=in_channels, act=self.act, with_conv=self.with_conv))
                f_res *= 2
            self.Upsampling.append(
                ResNetBlock(in_features=in_channels, out_features=out_channels, act=self.act)
            )
            if f_res in self.attn_resolutions:
                self.Upsampling.append(AttnBlock(features=out_channels))
        # Final
        self.Upsampling.append(
            nn.Sequential(
                Custumnn.Linear(out_channels, out_channels),
                nn.Sigmoid()
            ))

    def forward(self, x: Tensor) -> List[Tensor]:
        h = x
        # Encoder
        h, skip = self.encode(h)
        # Vector quantization
        h, vq_loss = self.vq_layer(h)
        # Decoder
        h = self.decode(h, skip)
        return [h, vq_loss]

    def encode(self, x):
        skip = []
        h = x
        # Down
        for module in self.Downsampling:
            h = module(h)
            skip.append(h)
        return h, skip

    def decode(self, x, skip):
        skip.reverse()
        h = x
        # Up
        for index, module in enumerate(self.Upsampling):
            if index < len(skip):
                h += skip[index]
            h = module(h)
            
        return h

    def loss_function(self, x, x_rec, vq_loss):
        
        rec_loss = F.mse_loss(x_rec, x)
        loss = rec_loss + vq_loss
        return loss

    def training_step(self, batch, batch_idx):
        x, _ = batch
        self.curr_device = x.device
        x_rec, vq_loss = self(x)
        loss = self.loss_function(x, x_rec, vq_loss)
        self.log('TL', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        self.curr_device = x.device
        x_rec, vq_loss = self(x)
        loss = self.loss_function(x, x_rec, vq_loss)
        self.log('VL', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        self.curr_device = x.device
        x_rec, vq_loss = self(x)
        loss = self.loss_function(x, x_rec, vq_loss)
        self.log('TeL', loss, on_step=False, on_epoch=True, sync_dist=True)

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
    
    def on_validation_epoch_end(self):
        if self.current_epoch % self.sampling_period == 0:
            self.sample_images()
    
    # def sample(self, num_samples):
        
    #     z = torch.randn(num_samples, self.latent_dim, device=self.curr_device)

    #     # Retoration
    #     z = self.FC_restore(z)
    #     h = z.view([-1, self.latent_ch, self.latent_res, self.latent_res])
        
    #     # Decoder
    #     h = self.decode(h)
    #     return h
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

        # try:
        #     samples = self.sample(16)
            
        #     samples_dir = os.path.join(self.logger.log_dir, "Samples")
        #     os.makedirs(samples_dir, exist_ok=True)

        #     vutils.save_image(samples.cpu().data,
        #                       os.path.join(samples_dir,
        #                                    f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
        #                       normalize=True,
        #                       nrow=4)
        # except Warning:
        #     pass
