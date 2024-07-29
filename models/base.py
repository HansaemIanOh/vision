from .flowers_vae import FLOWERVAEModel
from .flowers_cnn import FLOWERCNNModel

chose_model = {'flowers_vae':FLOWERVAEModel,
               'flowers_cnn':FLOWERCNNModel}

def get_model(name, config):
    return chose_model[name](config)