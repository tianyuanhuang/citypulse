"""
Code of *CityPulse: Fine-Grained Assessment of Urban Change with Street View Time Series*
Author: Tianyuan Huang, Zejia Wu, Jiajun Wu, Jackelyn Hwang and Ram Rajagopal

References:

MAE: *Masked Autoencoders Are Scalable Vision Learners*
     https://github.com/facebookresearch/mae
     Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll{\'a}r and Ross Girshick

STEGO: *STEGO: Unsupervised Semantic Segmentation by Distilling Feature Correspondences*
       https://github.com/mhamilton723/STEGO
       Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, William T. Freeman       
"""

import sys
import math
import random
from os.path import join
from tqdm import tqdm
import timm.optim.optim_factory as optim_factory

import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import torch.multiprocessing
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils import *
from modules import *
from data import Data150kIMG

import warnings
warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')

class MAEGSVModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        self.model = mae_vit(cfg)

        self.automatic_optimization = False
        self.softmax = nn.Softmax()

        self.val_steps = 0
        self.save_hyperparameters()

    def forward(self, x, year, month, flat=True):
        x, mask, ids_restore = self.model.forward_encoder(x, year, month, mask_ratio=0)
        x_ = x[:, 1:, :]
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1) # [N, 1+TL, 1024]
        if flat:
            return x.reshape(x.shape[0], -1)
        else:
            return x

    def adjust_learning_rate(self, optimizer):
        """Decay the learning rate with half-cycle cosine after warmup"""
        step = self.global_step
        if step < self.cfg.warmup_steps:
            lr = self.cfg.lr * step / self.cfg.warmup_steps 
        else:
            lr = self.cfg.min_lr + (self.cfg.lr - self.cfg.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (step - self.cfg.warmup_steps) / (self.cfg.max_steps - self.cfg.warmup_steps)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr

    def configure_optimizers(self):
        model_params = optim_factory.add_weight_decay(self.model, self.cfg.weight_decay)
        model_optim = torch.optim.AdamW(model_params, lr=self.cfg.lr, betas=(0.9, 0.95))
        return model_optim

    def training_step(self, batch, batch_idx):
        model_optim= self.optimizers()
        self.adjust_learning_rate(model_optim)
        model_optim.zero_grad()

        with torch.no_grad():
            ind = batch["ind"].to(self.device)
            seq_index = batch["seq_idx"]
            year = batch["year"].to(self.device)
            month = batch["month"].to(self.device)
            imgs = batch["img_building"].to(self.device)
            
        loss, pred, mask = self.model(imgs, year, month, mask_ratio=self.cfg.mask_ratio)

        total_loss = 0
        total_loss += loss

        log_args = dict(sync_dist=False, rank_zero_only=True)

        with torch.no_grad():
            self.log('loss/recon_loss', loss, **log_args)
            self.log('loss/total', total_loss, **log_args)

        self.manual_backward(loss)
        model_optim.step()

        if self.global_step % 2000 == 0 and self.global_step > 0:
            print("RESETTING TFEVENT FILE")
            # Make a new tfevent file
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()

        return total_loss

    def validation_step(self, batch, batch_idx):
        ind = batch["ind"].to(self.device)
        seq_index = batch["seq_idx"]
        year = batch["year"].to(self.device) # [BS, seq_len]
        month = batch["month"].to(self.device) # [BS, seq_len]
        imgs = batch["img_building"].to(self.device) # [BS, seq_len, 3, 224, 224]

        return {'imgs': imgs.detach().cpu(),
                "seq_idxs": seq_index,
                "years": year.detach().cpu(),
                "months": month.detach().cpu()}

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        with torch.no_grad():

            if self.trainer.is_global_zero:
                #output_num = 0
                output_num = random.randint(0, len(outputs) -1)
                output = {k: v.detach().cpu() for k, v in outputs[output_num].items() if k!="seq_idxs"}
                output["seq_idxs"] = outputs[output_num]["seq_idxs"]

                imgs = output["imgs"]
                seq_idxs = output["seq_idxs"]
                years = output["years"]
                months = output["months"]

                seq_sample = random.sample(range(0, imgs.shape[0]), self.cfg.n_seq)

                for seq in seq_sample:
                    img = imgs[seq].to(self.device)
                    seq_idx = seq_idxs[seq]
                    year = years[seq].to(self.device)
                    month = months[seq].to(self.device)
                    if img.shape[0] != self.cfg.seq_len:
                        n = img.shape[0]
                    else:
                        n = self.cfg.seq_len

                    with torch.no_grad():
                        latent, mask, ids_restore = self.model.forward_encoder(img.unsqueeze(0), year.unsqueeze(0), month.unsqueeze(0), self.cfg.mask_ratio)
                        pred = self.model.forward_decoder(latent, year.unsqueeze(0), month.unsqueeze(0), ids_restore) # [N, T*L, p*p*3]
                        
                        ori_img = img

                        img = self.model.patchify_seq(ori_img.unsqueeze(0))
                        img[:, mask[0].bool(), :] = 0.5
                        img = self.model.unpatchify_seq(img).cpu() # [1, T, 3, 224, 224]
                        
                        pred[:, (1-mask[0]).bool(), :] = 0
                        img_pred = self.model.patchify_seq(ori_img.unsqueeze(0))
                        img_pred[:, mask[0].bool(), :] = 0
                        pred = pred + img_pred
                        pred = self.model.unpatchify_seq(pred).cpu() # [1, T, 3, 224, 224]

                        ori_img = ori_img.unsqueeze(0).cpu()

                        fig, ax = plt.subplots(3, n, figsize=(n * 6, 3 * 6))
                        for i in range(n-1):
                            ax[0, i].set_title(seq_idx+"-"+str(year[i].item())+"-"+str(month[i].item()), fontsize=10)
                            ax[0, i].imshow(ori_img[0, i].permute(1,2,0))
                            ax[1, i].imshow(img[0, i].permute(1,2,0))
                            ax[2, i].imshow(pred[0, i].permute(1,2,0))
                        ax[0, 0].set_ylabel("Original_Image", fontsize=10)
                        ax[1, 0].set_ylabel("Masked_Image", fontsize=10)
                        ax[2, 0].set_ylabel("Reconstructed_Image", fontsize=10)

                        plt.tight_layout()
                        add_plot(self.logger.experiment, "Seq_IMAGEs", self.global_step)

@hydra.main(config_path="configs", config_name="mae_pretrain.yml")
def train_main(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))

    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    prefix = "{}/{}_{}".format(log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix
    print(name)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(seed=0)

    # print(cfg.output_root)
    sys.stdout.flush()

    train_dataset = Data150kIMG(cfg, if_seq=True, building_img_only=True, img_only=False)
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    val_dataset = Data150kIMG(cfg, if_seq=True, if_val=True, building_img_only=True, img_only=False)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    model = MAEGSVModel(cfg)

    tb_logger = TensorBoardLogger(
        name,
        default_hp_metric=False
    )

    gpu_args = dict(gpus=-1, accelerator='ddp', val_check_interval=cfg.val_freq)

    if gpu_args["val_check_interval"] > len(train_loader) // 4:
        gpu_args.pop("val_check_interval")

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        max_steps=cfg.max_steps,
        callbacks=[
            ModelCheckpoint(
                dirpath=join(checkpoint_dir, name.split("/")[-1]),
                every_n_train_steps=400,
                monitor=None,
            ),
            LearningRateMonitor(logging_interval='step')
        ],
        **gpu_args
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    train_main()