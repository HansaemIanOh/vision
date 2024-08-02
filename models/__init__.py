from .vae import VAEModel
from .cnn import CNNModel
from .ddpm import DDPMModel
from .vqvae import VQVAEModel
from .vqvaeunet import VQVAEUNETModel
from .gan import GANmodel
chose_model = {'vae':VAEModel,
               'cnn':CNNModel,
               'ddpm':DDPMModel,
               'vqvae':VQVAEModel,
               'vqvaeunet':VQVAEUNETModel,
               'gan':GANmodel
               }
def get_model(name, config):
    return chose_model[name](config)