import os
import yaml
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import seed_everything
from dataloader import ImageFolderDataModule
from models import get_model
from torchinfo import summary
from safetensors.torch import save_file, load_file
torch.set_float32_matmul_precision('high')

parser = argparse.ArgumentParser()
parser.add_argument('--config',
                    dest='filename',
                    default='configs/flowers_cnn.yaml')
args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

data_module = ImageFolderDataModule(**config['data_config'])

data_module.setup()

# TensorBoard logger
logger = TensorBoardLogger(save_dir=config['log_config']['save_dir'], name=config['log_config']['name'])

seed_everything(config['model_config']['manual_seed'], True)

model = get_model(config['model_config']['name'], config['model_config'])
if config['log_config']['checkpoint']:
    current_version = int(logger.log_dir.split('version_')[-1])
    prev_version_dir = os.path.join(os.path.dirname(logger.log_dir), f'version_{current_version-1}')
    last_ckpt = os.path.join(prev_version_dir, "checkpoints", "last.ckpt")
    model = model.load_from_checkpoint(last_ckpt, config=config['model_config']) if os.path.exists(last_ckpt) else model
    exit()

patch_size = config['model_config']['patch_size']

@rank_zero_only
def model_summary(model, input_size=(1, 3, patch_size, patch_size)):
    summary(model, input_size=(1, 3, patch_size, patch_size))
if config['log_config']['model_summary']:
    model_summary(model, input_size=(1, 3, patch_size, patch_size))

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(logger.log_dir , "checkpoints"),
    save_top_k=2,
    monitor='VL',
    save_last= True
)

trainer = pl.Trainer(
    logger=logger,
    callbacks=[LearningRateMonitor(),
               checkpoint_callback],
    log_every_n_steps=1,
    enable_progress_bar=True,
    **config['trainer_config']
)
trainer.fit(model, datamodule=data_module)

@rank_zero_only
def test_model():
    test_trainer = pl.Trainer(devices=1, num_nodes=1)
    test_trainer.test(model, data_module.test_dataloader(), verbose=True)

test_model()
