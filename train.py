import os
import hydra
import wandb
import shutil
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import WandbLogger

from src.utils.lightning import DataModule, CycleGAN_LightningSystem
from src.models.cycle_gan import CycleGAN_Unet_Generator, CycleGAN_Discriminator, CycleGAN_Resnet_Generator
from src.utils.utils import init_weights
from src.utils.transforms import ImageTransform


@hydra.main(config_path='.', config_name='config')
def main(cfg: DictConfig):
    print('Train CycleGAN Model')
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    seed_everything(cfg.train.seed)

    # Init asset dir  --------------------------------------------------
    try:
        # Remove checkpoint folder
        shutil.rmtree(cfg.data.asset_dir)
    except:
        pass

    os.makedirs(cfg.data.asset_dir, exist_ok=True)

    # Logger  --------------------------------------------------
    wandb.login()
    logger = WandbLogger(project='AI-Painter', reinit=True)
    logger.log_hyperparams(dict(cfg.data))
    logger.log_hyperparams(dict(cfg.train))

    # Transforms  --------------------------------------------------
    transform = ImageTransform(cfg.data.img_size)

    # DataModule  --------------------------------------------------
    dm = DataModule(cfg, transform, phase='train')

    # Model Networks  --------------------------------------------------
    nets = {
        'G_basestyle': init_weights(CycleGAN_Unet_Generator(), init_type='normal'),
        'G_stylebase': init_weights(CycleGAN_Unet_Generator(), init_type='normal'),
        'D_base': init_weights(CycleGAN_Discriminator(), init_type='normal'),
        'D_style': init_weights(CycleGAN_Discriminator(), init_type='normal'),
    }

    # Lightning System  --------------------------------------------------
    model = CycleGAN_LightningSystem(cfg, transform, **nets)

    # Train  --------------------------------------------------
    trainer = Trainer(
        logger=logger,
        max_epochs=cfg.train.epoch,
        gpus=1,
        reload_dataloaders_every_epoch=True,
        num_sanity_val_steps=0,  # Skip Sanity Check
    )

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
