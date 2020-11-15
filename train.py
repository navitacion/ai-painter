import os
import hydra
from omegaconf import DictConfig
from comet_ml import Experiment
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.lightning import MonetDataModule, CycleGAN_LightningSystem, VanGoghDataModule
from src.models.cycle_gan import CycleGAN_Unet_Generator, CycleGAN_Discriminator
from src.utils.utils import ImageTransform, seed_everything, init_weights


@hydra.main('config.yml')
def main(cfg: DictConfig):
    cur_dir = hydra.utils.get_original_cwd()
    os.chdir(cur_dir)
    # Config  ###################################################################
    api_key = cfg.comet_ml.api_key
    project_name = cfg.comet_ml.project_name
    data_dir = './data/'
    checkpoint_path = './checkpoints'
    transform = ImageTransform(img_size=256)
    batch_size = 1
    lr = {
        'G': 0.0002,
        'D': 0.0002
    }
    epoch = 500
    seed = 42
    reconstr_w = 10
    id_w = 5

    seed_everything(seed)

    # Comet_ml
    experiment = Experiment(api_key=api_key,
                            project_name=project_name)

    if cfg.train.style == 'monet':
        dm = MonetDataModule(data_dir, transform, batch_size, phase='train', seed=seed)

    elif cfg.train.style == 'vangogh':
        dm = VanGoghDataModule(data_dir, transform, batch_size, phase='train', seed=seed)
    else:
        raise TypeError("Please enter the correct arguments 'train.style' ")

    G_basestyle = CycleGAN_Unet_Generator()
    G_stylebase = CycleGAN_Unet_Generator()
    D_base = CycleGAN_Discriminator()
    D_style = CycleGAN_Discriminator()

    # Init Weight
    for net in [G_basestyle, G_stylebase, D_base, D_style]:
        init_weights(net, init_type='normal')

    model = CycleGAN_LightningSystem(G_basestyle, G_stylebase, D_base, D_style,
                                     lr, transform, experiment, reconstr_w, id_w, checkpoint_path)


    checkpoint_callback = ModelCheckpoint(
        filepath='./checkpoints',
        save_top_k=-1,
        verbose=False,
        mode='min',
        prefix=f'cyclegan'
    )

    trainer = Trainer(
        logger=False,
        max_epochs=epoch,
        gpus=1,
        checkpoint_callback=checkpoint_callback,
        reload_dataloaders_every_epoch=True,
        num_sanity_val_steps=0,  # Skip Sanity Check
    )

    # Train
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    main()
