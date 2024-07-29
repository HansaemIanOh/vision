import math
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

class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
    def forward(self, x: Tensor) -> Tensor:
        
        expansion = False
        if len(x.shape) > 2:
            expansion = True
        if expansion:
            order, inverse_order = self.reorder_dimensions(len(x.shape))
            x = x.permute(order).contiguous()
        x = self.linear(x)
        if expansion:
            x = x.permute(inverse_order).contiguous()
        return x
    @staticmethod
    def reorder_dimensions(shape_len):
        order = list(range(shape_len))
        order = [0] + order[2:] + [1]
        inverse_order = [order.index(i) for i in range(shape_len)]
        return order, inverse_order

class DownBlock(nn.Module):
    def __init__(self, in_features, out_features, act, with_conv=True):
        super().__init__()
        self.conv2d_type2 = nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=2, padding=1) # type1 : 311, type2: 3,2,1
        self.pool = nn.AvgPool2d((2, 2))
        self.act = act
        self.with_conv = with_conv

    def forward(self, x : Tensor, temb : Tensor) -> Tensor:
        
        if self.with_conv:
            x = self.conv2d_type2(x)
        else:
            x = self.pool(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_features, out_features, act, with_conv=True):
        super().__init__()
        self.conv2d_type1 = nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=1, padding=1)
        self.interpolate = nn.Upsample(scale_factor=2, mode='nearest')
        self.act = act
        self.with_conv = with_conv

    def forward(self, x : Tensor, temb : Tensor) -> Tensor:
        
        x = self.interpolate(x)
        if self.with_conv:
            x = self.conv2d_type1(x)
        return x

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, act, time_features, conv_shortcut=False, dropout=0.):
        super().__init__()
        self.act = act

        self.conv2d_1 = nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_2 = nn.Conv2d(out_features, out_features, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2d_3 = nn.Conv2d(in_features, out_features, kernel_size=(3, 3), stride=1, padding=1)

        self.norm1 = nn.GroupNorm(num_groups=in_features, num_channels=in_features)
        self.norm2 = nn.GroupNorm(num_groups=out_features, num_channels=out_features)

        self.linear_1 = Custumnn.Linear(time_features, out_features)
        self.linear_2 = Custumnn.Linear(in_features, out_features)
        self.dropout = dropout
        self.conv_shortcut = conv_shortcut
        self.dropout_nn = nn.Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.time_features = time_features
    def forward(self, x : Tensor, temb=None) -> Tensor:
        if temb==None:
            temb = torch.randint(0, 1000, (x.shape[0], 64), device=x.device, dtype=x.dtype)
        h = x
        h = self.act(self.norm1(h))
        h = self.conv2d_1(h)
        temb = self.linear_1(self.act(temb))
        h+= temb.view(match_dims(temb, h))
        h = self.act(self.norm2(h))
        h = self.dropout_nn(h)
        h = self.conv2d_2(h)

        if self.in_features!= self.out_features:
            
            if self.conv_shortcut:
                x = self.conv2d_3(x) # out_features
            else:
                x = self.linear_2(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.linear1 = Custumnn.Linear(features, features)
        self.linear2 = Custumnn.Linear(features, features)
        self.norm = nn.GroupNorm(num_groups=features, num_channels=features)

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

class UNet(nn.Module):
    
    def __init__(self,
                 *,
                 ch,
                 out_ch,
                 ch_mult = (1, 1, 2, 2, 4, 4),
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.,
                 resamp_with_conv = True,
                 img_size,
                 **kwargs):
        super().__init__()
        self.act = nn.GELU()
        self.linear = Custumnn.Linear
        self.conv2d = nn.Conv2d
        self.down = DownBlock
        self.up = UpBlock
        self.resnet = ResNetBlock
        self.attn = AttnBlock

        self.ch = ch
        self.out_ch = out_ch
        self.ch_mult = ch_mult
        self.num_res_blocks = num_res_blocks
        self.attn_resolutions = attn_resolutions
        self.dropout = dropout
        self.resamp_with_conv = resamp_with_conv

        self.linear_1 = self.linear(self.ch, self.ch * 4)
        self.linear_2 = self.linear(self.ch * 4, self.ch * 4)
        self.time_features = self.ch * 4
        
        num_resolutions = len(self.ch_mult)

        assert type(self.ch) == int, "channel type is not int!!"
        self.start = self.conv2d(3, self.ch, kernel_size=(3, 3), stride=1, padding=1)
        # Downsampling
        # self.Downsampling = []
        self.Downsampling = nn.ModuleList()
        in_features = self.ch
        f_res = img_size
        for i_level in range(num_resolutions):
            for i_block in range(self.num_res_blocks):
                self.Downsampling.append(
                    self.resnet(in_features=in_features, out_features=self.ch * self.ch_mult[i_level], 
                                act=self.act, time_features=self.time_features, dropout=self.dropout,
                                )
                )
                in_features = self.ch * self.ch_mult[i_level]
                if f_res in self.attn_resolutions:
                    self.Downsampling.append(self.attn(in_features, in_features))
            # Downsample
            if i_level != num_resolutions - 1:
                self.Downsampling.append(self.down(in_features, in_features, act=self.act, with_conv=self.resamp_with_conv))
                f_res /= 2
        # Middle
        self.Middle = nn.ModuleList()
        self.Middle.append(self.resnet(in_features, in_features, self.act, self.time_features, dropout=self.dropout))
        self.Middle.append(self.attn(in_features))
        self.Middle.append(self.resnet(in_features, in_features, self.act, self.time_features, dropout=self.dropout))

        # Upsampling
        self.Upsampling = nn.ModuleList()
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                self.Upsampling.append(
                    self.resnet(in_features=in_features, out_features=self.ch * self.ch_mult[i_level], 
                                act=self.act, time_features=self.time_features, dropout=self.dropout, 
                                )
                )
                in_features = self.ch * self.ch_mult[i_level]
                if f_res in self.attn_resolutions:
                    self.Upsampling.append(self.attn(in_features, in_features))
            # Upsample
            if i_level != 0:
                self.Upsampling.append(self.up(in_features, in_features, act=self.act, with_conv=self.resamp_with_conv))
                f_res *= 2
        # End
        self.End = nn.ModuleList()
        self.End.append(nn.Sequential(
            nn.GroupNorm(num_groups=in_features, num_channels=in_features),
            self.act,
            self.conv2d(in_features, self.out_ch, kernel_size=(3, 3), stride=1, padding=1)
        ))
    def forward(self, x : Tensor, t= None) -> Tensor:
        
        if t == None:
            t = torch.randint(low=0, high=1000, size=(x.shape[0],), device=x.device)
        h = x
        # Timestep embedding
        temb = timeembed(t, self.ch)
        temb = self.linear_1(temb)
        temb = self.linear_2(self.act(temb))

        # Start

        h = self.start(h)
        
        # Downsampling

        for module in self.Downsampling:
            h = module(h, temb)

        # Middle

        for module in self.Middle:
            h = module(h, temb)

        # Upsampling

        for module in self.Upsampling:
            h = module(h, temb)

        # End

        for module in self.End:
            h = module(h)
        return h

class DDPM_Model(pl.LightningModule):
    
    def __init__(self,
                 t_max=1000,
                 *,
                 ch,
                 out_ch,
                 ch_mult = (1, 2, 4, 8),
                 num_res_blocks,
                 attn_resolutions,
                 dropout=0.,
                 resamp_with_conv = True,
                 img_size,
                 latent_dim,
                 **kwargs) -> None:
        super().__init__()
        self.t_max = t_max
        self.latent_dim = latent_dim
        self.model = UNet(ch=ch, 
                          out_ch=out_ch, 
                          ch_mult=ch_mult, 
                          num_res_blocks=num_res_blocks, 
                          attn_resolutions=attn_resolutions, 
                          dropout=dropout, 
                          resamp_with_conv=resamp_with_conv,
                          img_size=img_size)

    def forward(self, 
                input : Tensor, 
                t=None, 
                **kwargs) -> List[Tensor]:
        return self.model(input, t)    
        # # assert t != None, "T is none type" T 스케쥴링 꼭 해야함. 안하면 훈련이 안됨.

        # if t==None:
        #     print("time is None")
        #     t = torch.randint(0, 1000, (input.shape[0],), device=input.device)
        # eps_theta = self.model(input, t=t) # UNet

        # return [eps_theta, input, t]

    # def loss_function(self,
    #                   *args,
    #                   **kwargs) -> dict:
    #     # [eps_theta, input]
    #     x_start = args[1]
    #     t = args[2]
    #     loss = self.p_losses(x_start=x_start, t=t, model=self.model) # UNet

    #     return {'loss': loss}
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
        self.log('Val Loss', loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        x_rec, mu, logvar = self(x)
        rec_loss = F.mse_loss(x_rec, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        loss = rec_loss + self.kld_weight * kld_loss
        self.log('Test Loss', loss, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config['learning_rate'])
    
    def on_validation_end(self) -> None:
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
    def loss_function(self,
                      x_start,
                      **kwargs) -> dict:
        # [eps_theta, input]
        
        loss = self.inner_loop(x_start)

        return {'loss': loss}
    
    def t_sample(self, x_start, low, high):
        return torch.arange(low, high+1, device=x_start.device)

    def inner_loop(self, x_0): # x_0만 주면 알아서 loss 전부 계산해서 반환하는 형식임.
        
        t_max = self.t_max
        t_min = 0

        inner_batch = 20
        x_0_repeat = x_0.repeat(inner_batch, 1, 1, 1)

        loss = 0
        # for cur_time in range(t_min, t_max, inner_batch):
        #     t = self.t_sample(x_0, cur_time+1, cur_time+inner_batch)
            # t_expanded = t.view(match_dims(t, x_0_repeat))
            # print("="*50)
            # print(x_0_repeat.shape, t_expanded.shape)
            # print("="*50)
            # exit()
            # loss += self.p_losses(x_start=x_0_repeat, t=t, model=self.model)
        t = torch.randint(t_min, t_max, (x_0_repeat.shape[0],), device=x_0_repeat.device)
        loss = self.p_losses(x_start=x_0_repeat, t=t, model=self.model)
        return loss

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
        
        c_0 = (torch.sqrt(alpha_bar_t_1) * beta) / (torch.sqrt(1 - alpha_bar_t))
        c_t = (torch.sqrt(alpha_bar_t) * (1 - alpha_bar_t_1)) / (torch.sqrt(1 - alpha_bar_t))
        
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
        # losses = (noise - x_recon).flatten(1).pow(2).mean(1)
        
        losses = F.mse_loss(eps, eps_theta)

        # assert losses.shape == x_start.shape[0]
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
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        x_T = torch.randn([num_samples] + self.latent_dim, device=current_device)
        x_t = x_T
        T = self.t_max
        t_tensor = torch.ones((x_T.shape[0],), device=current_device) * T

        t=1000
        while t>=1:
            x_t_1 = self.p_sample(x_t, t_tensor, model=self.model)
            t_tensor -= 1
            t-=1
        x_1 = x_t_1
        # x_0 = self.p_final(x_1)
        x_0 = x_1
        return x_0
    
