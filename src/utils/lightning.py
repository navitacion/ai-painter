import os, random, time, cv2
import numpy as np
from pathlib import Path
import wandb

import torch
from torchvision.utils import make_grid
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .dataset import CycleGanDataset

# DataModule ---------------------------------------------------------------------------
class DataModule(pl.LightningDataModule):
    def __init__(self, cfg, transform, phase='train'):
        super(DataModule, self).__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.data.data_dir)
        self.transform = transform
        self.phase = phase

    def prepare_data(self):
        self.style_img_paths = [str(path) for path in self.data_dir.glob(f'{self.cfg.data.style_img_folder}/**/*.jpg')]
        self.base_img_paths = [str(path) for path in self.data_dir.glob(f'{self.cfg.data.base_img_folder}/**/*.jpg')]


    def _shuffle_imgpaths(self):
        random.seed()
        random.shuffle(self.base_img_paths)
        random.shuffle(self.style_img_paths)
        random.seed(self.cfg.train.seed)

    def train_dataloader(self):
        self._shuffle_imgpaths()
        self.train_dataset = CycleGanDataset(self.base_img_paths, self.style_img_paths, self.transform, self.phase)

        return DataLoader(self.train_dataset,
                          batch_size=self.cfg.train.batch_size,
                          shuffle=True,
                          num_workers=self.cfg.train.num_workers,
                          pin_memory=False
                          )


# CycleGAN - Lightning Module ---------------------------------------------------------------------------
class CycleGAN_LightningSystem(pl.LightningModule):
    def __init__(self, cfg, transform, G_basestyle, G_stylebase, D_base, D_style):
        super(CycleGAN_LightningSystem, self).__init__()
        self.cfg = cfg
        self.G_basestyle = G_basestyle
        self.G_stylebase = G_stylebase
        self.D_base = D_base
        self.D_style = D_style
        self.lr = dict(cfg.train.lr)
        self.transform = transform
        self.reconstr_w = cfg.train.reconstr_w
        self.identity_w = cfg.train.identity_w
        self.checkpoint_path = cfg.data.asset_dir

        # Loss_fn
        self.mae = nn.L1Loss()
        self.generator_loss = nn.MSELoss()
        self.discriminator_loss = nn.MSELoss()

    def configure_optimizers(self):
        self.g_basestyle_optimizer = optim.Adam(self.G_basestyle.parameters(), lr=self.lr['G'], betas=(0.5, 0.999))
        self.g_stylebase_optimizer = optim.Adam(self.G_stylebase.parameters(), lr=self.lr['G'], betas=(0.5, 0.999))
        self.d_base_optimizer = optim.Adam(self.D_base.parameters(), lr=self.lr['D'], betas=(0.5, 0.999))
        self.d_style_optimizer = optim.Adam(self.D_style.parameters(), lr=self.lr['D'], betas=(0.5, 0.999))

        return [self.g_basestyle_optimizer, self.g_stylebase_optimizer, self.d_base_optimizer,
                self.d_style_optimizer], []

    def training_step(self, batch, batch_idx, optimizer_idx):
        base_img, style_img = batch
        b = base_img.size()[0]

        valid = torch.ones(b, 1, 30, 30).cuda()
        fake = torch.zeros(b, 1, 30, 30).cuda()

        # Train Generator
        if optimizer_idx == 0 or optimizer_idx == 1:
            # Validity
            # MSELoss
            val_base = self.generator_loss(self.D_base(self.G_stylebase(style_img)), valid)
            val_style = self.generator_loss(self.D_style(self.G_basestyle(base_img)), valid)
            val_loss = (val_base + val_style) / 2

            # Reconstruction
            reconstr_base = self.mae(self.G_stylebase(self.G_basestyle(base_img)), base_img)
            reconstr_style = self.mae(self.G_basestyle(self.G_stylebase(style_img)), style_img)
            reconstr_loss = (reconstr_base + reconstr_style) / 2

            # Identity
            identity_base = self.mae(self.G_stylebase(base_img), base_img)
            identity_style = self.mae(self.G_basestyle(style_img), style_img)
            identity_loss = (identity_base + identity_style) / 2

            # Loss Weight
            G_loss = val_loss + self.reconstr_w * reconstr_loss + self.identity_w * identity_loss

            logs = {'loss': G_loss, 'validity': val_loss.detach(), 'reconstr': reconstr_loss.detach(), 'identity': identity_loss.detach()}
            self.log(f'train/G_loss', G_loss, on_epoch=True)
            self.log(f'train/validity', val_loss, on_epoch=True)
            self.log(f'train/reconstr', reconstr_loss, on_epoch=True)
            self.log(f'train/identity', identity_loss, on_epoch=True)

            return logs

        # Train Discriminator
        elif optimizer_idx == 2 or optimizer_idx == 3:
            # MSELoss
            D_base_gen_loss = self.discriminator_loss(self.D_base(self.G_stylebase(style_img)), fake)
            D_style_gen_loss = self.discriminator_loss(self.D_style(self.G_basestyle(base_img)), fake)
            D_base_valid_loss = self.discriminator_loss(self.D_base(base_img), valid)
            D_style_valid_loss = self.discriminator_loss(self.D_style(style_img), valid)

            D_gen_loss = (D_base_gen_loss + D_style_gen_loss) / 2

            # Loss Weight
            D_loss = (D_gen_loss + D_base_valid_loss + D_style_valid_loss) / 3

            logs = {'loss': D_loss}
            self.log(f'train/D_loss', D_loss, on_epoch=True)

            return logs

    def training_epoch_end(self, outputs):
        if self.current_epoch % 10 == 0:
            # Display Model Output
            data_dir = Path(self.cfg.data.data_dir)
            target_img_paths = [str(path) for path in data_dir.glob(f'{self.cfg.data.base_img_folder}/**/*.jpg')][:8]

            target_imgs = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in target_img_paths]
            target_imgs = [self.transform(img, phase='test') for img in target_imgs]
            target_imgs = torch.stack(target_imgs, dim=0)
            target_imgs = target_imgs.cuda()

            gen_imgs = self.G_basestyle(target_imgs)
            gen_img = torch.cat([target_imgs, gen_imgs], dim=0)

            # Reverse Normalization
            gen_img = gen_img * 0.5 + 0.5
            gen_img = gen_img * 255

            joined_images_tensor = make_grid(gen_img, nrow=4, padding=2)

            joined_images = joined_images_tensor.detach().cpu().numpy().astype(int)
            joined_images = np.transpose(joined_images, [1, 2, 0])

            wandb.log({"output_img": [wandb.Image(joined_images, caption=f'Epoch {self.current_epoch}')]})

            # Save checkpoints
            if self.checkpoint_path is not None:
                model = self.G_basestyle
                weight_name = f'{self.cfg.data.style_img_folder}_weight_epoch_{self.current_epoch}.pth'
                weight_path = os.path.join(self.checkpoint_path, weight_name)
                torch.save(model.state_dict(), weight_path)
                wandb.save(weight_path)
                time.sleep(3)
        else:
            pass

        return None