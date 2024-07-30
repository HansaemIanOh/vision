from .flowers_vae import FLOWERVAEModel
from .flowers_cnn import FLOWERCNNModel
from .flowers_ddpm import FLOWERDDPMModel

chose_model = {'flowers_vae':FLOWERVAEModel,
               'flowers_cnn':FLOWERCNNModel,
               'flowers_ddpm':FLOWERDDPMModel}

def get_model(name, config):
    return chose_model[name](config)