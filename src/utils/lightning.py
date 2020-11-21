import os, glob, random, time
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision.utils import make_grid
from torch import nn, optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .utils import CycleGanDataset

# DataModule ---------------------------------------------------------------------------
class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, style_img_dir, transform, batch_size, phase='train', seed=0):
        super(MonetDataModule, self).__init__()
        self.data_dir = data_dir
        self.style_img_dir = style_img_dir
        self.transform = transform
        self.batch_size = batch_size
        self.phase = phase
        self.seed = seed

    def prepare_data(self):
        self.base_img_paths = glob.glob(os.path.join(self.data_dir, 'photo_jpg', '*.jpg'))
        self.style_img_paths = glob.glob(os.path.join(self.style_img_dir, '*.jpg'))

    def train_dataloader(self):
        random.seed()
        random.shuffle(self.base_img_paths)
        random.shuffle(self.style_img_paths)
        random.seed(self.seed)
        self.train_dataset = CycleGanDataset(self.base_img_paths, self.style_img_paths, self.transform, self.phase)

        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4,
                          pin_memory=True
                          )


# CycleGAN - Lightning Module ---------------------------------------------------------------------------
class CycleGAN_LightningSystem(pl.LightningModule):
    def __init__(self, G_basestyle, G_stylebase, D_base, D_style, lr, transform, experiment,
                 reconstr_w=10, id_w=2, checkpoint_path=None):
        super(CycleGAN_LightningSystem, self).__init__()
        self.G_basestyle = G_basestyle
        self.G_stylebase = G_stylebase
        self.D_base = D_base
        self.D_style = D_style
        self.lr = lr
        self.transform = transform
        self.reconstr_w = reconstr_w
        self.id_w = id_w
        self.experiment = experiment
        self.checkpoint_path = checkpoint_path
        self.cnt_train_step = 0
        self.cnt_epoch = 0

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

        # Count up
        self.cnt_train_step += 1

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
            id_base = self.mae(self.G_stylebase(base_img), base_img)
            id_style = self.mae(self.G_basestyle(style_img), style_img)
            id_loss = (id_base + id_style) / 2

            # Loss Weight
            G_loss = val_loss + self.reconstr_w * reconstr_loss + self.id_w * id_loss

            logs = {'loss': G_loss, 'validity': val_loss, 'reconstr': reconstr_loss, 'identity': id_loss}
            # self.experiment.log_metrics(logs, step=self.cnt_train_step)

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
            # self.experiment.log_metric('D_loss', D_loss, step=self.cnt_train_step)

            return logs

    def training_epoch_end(self, outputs):
        self.cnt_epoch += 1

        avg_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().detach() / 4 for i in range(4)])
        G_mean_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().detach() / 2 for i in [0, 1]])
        D_mean_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().detach() / 2 for i in [2, 3]])
        validity = sum([torch.stack([x['validity'] for x in outputs[i]]).mean().detach() / 2 for i in [0, 1]])
        reconstr = sum([torch.stack([x['reconstr'] for x in outputs[i]]).mean().detach() / 2 for i in [0, 1]])
        identity = sum([torch.stack([x['identity'] for x in outputs[i]]).mean().detach() / 2 for i in [0, 1]])

        logs = {
            'avg_loss': avg_loss, 'G_mean_loss': G_mean_loss, 'D_mean_loss': D_mean_loss,
            'validity': validity, 'reconstr': reconstr, 'identity': identity
        }

        self.experiment.log_metrics(logs, epoch=self.cnt_epoch)

        if self.cnt_epoch % 10 == 0:
            # Display Model Output
            target_img_paths = glob.glob('./data/photo_jpg/*.jpg')[:8]
            target_imgs = [self.transform(Image.open(path), phase='test') for path in target_img_paths]
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

            self.experiment.log_image(joined_images, name='output_img', step=self.cnt_epoch, image_channels='last')

            # Save checkpoints
            if self.checkpoint_path is not None:
                model = self.G_basestyle
                weight_name = f'weight_epoch_{self.cnt_epoch}.pth'
                weight_path = os.path.join(self.checkpoint_path, weight_name)
                torch.save(model.state_dict(), weight_path)
                time.sleep(3)
                self.experiment.log_asset(file_data=weight_path)
                os.remove(weight_path)
        else:
            pass

        return None