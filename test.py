

from utils.setup_configs import load_config
from utils.cifar10 import Cifar10DataModule
from models.lit_resnet import LitResnet
from pytorch_lightning import Trainer
from pytorch_lightning import Trainer
from utils.cifar10 import Cifar10DataModule
from utils.setup_configs import load_config
from models.lit_resnet import LitResnet
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything



if __name__ == '__main__':
    configs = load_config()

    seed_everything(configs[2]['logging_params']['manual_seed'])

    model = LitResnet.load_from_checkpoint('results/CIFAR10_LitResnet/3jmxcq8l/checkpoints/results/checkpoint/epoch=26-valid_loss=0.0000-valid_f1=0.0000.ckpt')

    wandb_logger = WandbLogger(
        project='CIFAR10_LitResnet', 
        name=configs[0]['model_params']['name'], 
        log_model=all,
        save_dir = 'results/'
    )

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        TQDMProgressBar(refresh_rate=10),
        ModelCheckpoint(
            save_top_k=1, 
            verbose=False, 
            save_last=True,
            save_weights_only=True,
            filename = 'results/checkpoint/{epoch:02d}-{valid_loss:.4f}-{valid_f1:.4f}',
            monitor='val_loss', 
            mode='min'
        ),
    ]

    trainer = Trainer(
        max_epochs = configs[1]['exp_params']['max_epochs'],
        accelerator = configs[1]['exp_params']['accelerator'],
        devices = configs[1]['exp_params']['devices'],
        logger = wandb_logger,
        callbacks = callbacks,
    )

    data = Cifar10DataModule()
    trainer.test(model, data)