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
from models.base import get_model
from torchinfo import summary
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
checkpoint = config['log_config']['checkpoint_path']
if checkpoint != 'None':
    model.load_from_checkpoint(checkpoint, config=config['model_config'])

@rank_zero_only
def model_summary(model, input_size=(1, 3, 224, 224)):
    summary(model, input_size=(1, 3, 224, 224))
if config['log_config']['model_summary']:
    model_summary(model, input_size=(1, 3, 224, 224))

# Checkpoint callback
checkpoint_callback = ModelCheckpoint(
    dirpath='logs/'+config['log_config']['name']+'/'+'checkpoint',
    filename=config['log_config']['name']+'_'+'{VL:.5f}',
    save_top_k=1,
    monitor='VL'
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
