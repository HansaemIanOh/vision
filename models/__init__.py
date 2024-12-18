from .vae import VAEModel
from .Bayesianvae import BVAEModel
from .cnnvae import CNNVAE
from .cnnae import CNNAE
from .cnn import CNNModel
from .ddpm import DDPMModel
from .vqvae import VQVAEModel
from .vqvaeunet import VQVAEUNETModel
from .gan import GANmodel
from .cgan import ConditionalGANmodel
from .resnet import ResNetModel
from .ldm import LatentDiffusion
chose_model = {'vae':VAEModel,
               'Bvae':BVAEModel,
               'cnnvae':CNNVAE,
               'cnnae':CNNAE,
               'cnn':CNNModel,
               'ddpm':DDPMModel,
               'vqvae':VQVAEModel,
               'vqvaeunet':VQVAEUNETModel,
               'gan':GANmodel,
               'resnet':ResNetModel,
               'cgan':ConditionalGANmodel
               }
def get_model(name, config):
    return chose_model[name](config)
