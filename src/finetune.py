"""
Code of *CityPulse: Fine-Grained Assessment of Urban Change with Street View Time Series*
Author: Tianyuan Huang, Zejia Wu, Jiajun Wu, Jackelyn Hwang and Ram Rajagopal

References:

ResNet: *Deep Residual Learning for Image Recognition*
        Kaiming He and Xiangyu Zhang and Shaoqing Ren and Jian Sun

MAE: *Masked Autoencoders Are Scalable Vision Learners*
     https://github.com/facebookresearch/mae
     Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll{\'a}r and Ross Girshick

CLIP: *Learning Transferable Visual Models From Natural Language Supervision*
      https://github.com/openai/CLIP
      Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever

BYOL: *Bootstrap your own latent: A new approach to self-supervised Learning*
      https://github.com/google-deepmind/deepmind-research/tree/master/byol
      Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko

DINO: *Emerging Properties in Self-Supervised Vision Transformers*
      https://github.com/facebookresearch/dino
      Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin

DINOv2: *DINOv2: Learning Robust Visual Features without Supervision*
        https://github.com/facebookresearch/dinov2
        Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski

STEGO: *STEGO: Unsupervised Semantic Segmentation by Distilling Feature Correspondences*
       https://github.com/mhamilton723/STEGO
       Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, William T. Freeman       
"""

import os
import sys
import random
from os.path import join

from tqdm import tqdm
from datetime import datetime
import argparse

import clip
import hydra
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from ignite.metrics import Accuracy, Precision, Recall

import torch
import torchvision
import torch.nn as nn
import torch.multiprocessing
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils import *
from modules import *
from byol_pretrain import BYOLGSVModel
from mae_pretrain import MAEGSVModel
from data import TestData20KScale

import warnings
warnings.filterwarnings("ignore")

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class SiamModel(nn.Module):
    def __init__(self, cfg) -> None:
        super(SiamModel, self).__init__()

        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.cfg.model == "ResNet101":
            input_dim = 1000
            self.head = torch.nn.Linear(input_dim*3, 1, bias=True)
            self.initialize_weights()
            self.model = torchvision.models.resnet101(pretrained=True)
        elif self.cfg.model == "DINO":
            input_dim = 768
            self.head = torch.nn.Linear(input_dim*3, 1, bias=True)
            self.initialize_weights()
            self.model = DinoFeaturizer(100, self.cfg)
        elif self.cfg.model == "CLIP":
            input_dim = 512
            self.head = torch.nn.Linear(input_dim*3, 1, bias=True)
            self.initialize_weights()
            self.model_name = cfg.model_type + "/" + str(cfg.dino_patch_size)
            self.model = clip.load(self.model_name, device=self.device, jit=False)[0].float()
        elif self.cfg.model == "DINOv2":
            input_dim = 768
            self.head = torch.nn.Linear(input_dim*3, 1, bias=True)
            self.initialize_weights()
            self.model_name = "dinov2" + "_" + cfg.model_type + str(cfg.dino_patch_size)
            self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        elif self.cfg.model == "StreetBYOL":
            input_dim = 768
            self.head = torch.nn.Linear(input_dim*3, 1, bias=True)
            self.initialize_weights()
            self.pretrained_model = BYOLGSVModel.load_from_checkpoint(self.cfg.ckpt)
            self.model = DinoFeaturizer(100, self.cfg)
            for m, n in zip(self.model.model.parameters(), self.pretrained_model.online_network.encoder.model.parameters()):
                m.data.copy_(n.data)
            del self.pretrained_model, m, n
        elif self.cfg.model == "StreetMAE":
            input_dim = 100864
            self.head = torch.nn.Linear(input_dim*3, 1, bias=True)
            self.initialize_weights() 
            self.model = MAEGSVModel.load_from_checkpoint(self.cfg.ckpt)
        else:
            raise ValueError("Unknown model.")

        if not self.cfg.finetune:
            if self.cfg.model in ["DINO", "StreetBYOL"]:
                for p in self.model.model.parameters():
                    p.requires_grad = False
                self.model.model.eval()
            else:
                for p in self.model.parameters():
                    p.requires_grad = False
                self.model.eval()
        else:
            for p in self.model.parameters():
                p.requires_grad = True
            self.model.train()

    def forward(self, x1, x2, **kw):
        if len(x1.shape) == 5:
            x1 = x1.squeeze(1)
        if len(x2.shape) == 5:
            x2 = x2.squeeze(1)
        if self.cfg.model in ["DINO", "StreetBYOL"]:
            embed_vec_early = self.model(x1, return_class_feat=True).reshape(x1.shape[0], -1)
            embed_vec_late = self.model(x2, return_class_feat=True).reshape(x2.shape[0], -1)
            x = torch.concat([embed_vec_early, embed_vec_late, embed_vec_late-embed_vec_early], dim=1)
        elif self.cfg.model == "ResNet101":
            embed_vec_early = self.model(x1).reshape(x1.shape[0], -1)
            embed_vec_late = self.model(x2).reshape(x2.shape[0], -1)
            x = torch.concat([embed_vec_early, embed_vec_late, embed_vec_late-embed_vec_early], dim=1)
        elif self.cfg.model == "CLIP":
            embed_vec_early = self.model.encode_image(x1).reshape(x1.shape[0], -1)
            embed_vec_late = self.model.encode_image(x2).reshape(x2.shape[0], -1)
            x = torch.concat([embed_vec_early, embed_vec_late, embed_vec_late-embed_vec_early], dim=1).float()
        elif self.cfg.model == "DINOv2":
            embed_vec_early = self.model(x1).reshape(x1.shape[0], -1)
            embed_vec_late = self.model(x2).reshape(x2.shape[0], -1)
            x = torch.concat([embed_vec_early, embed_vec_late, embed_vec_late-embed_vec_early], dim=1)
        elif self.cfg.model == "StreetMAE":
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
            year1 = year1.unsqueeze(1)
            year2 = year2.unsqueeze(1)
            month1 = month1.unsqueeze(1)
            month2 = month2.unsqueeze(1)
            embed_vec_early = self.model(x1, year1, month1, flat=True) # [N, (1+TL)*1024]
            embed_vec_late = self.model(x2, year2, month2, flat=True) # [N, (1+TL)*1024]
            x = torch.concat([embed_vec_early, embed_vec_late, embed_vec_late-embed_vec_early], dim=1)
        preds = self.head(x)
        return preds

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def reset_linear_parameters(self):
        with torch.no_grad():
            self.initialize_weights()

class Train_SiamModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.net = SiamModel(self.cfg)
        print(get_parameter_number(self.net))
        
        self.net_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.sigmoid = nn.Sigmoid()

        self.val_acc = Accuracy()
        self.val_recall = Recall()
        self.val_precision = Precision()

        self.save_hyperparameters()

    def forward(self, x1, x2, **kw):
        return  self.net(x1, x2, **kw)

    def configure_optimizers(self):
        net_params = [param for param in list(self.net.parameters()) if param.requires_grad==True]
        net_optim = torch.optim.Adam(net_params, lr=self.cfg.net_lr)
        return net_optim

    def training_step(self, batch, batch_idx):
        net_optim = self.optimizers()
        net_optim.zero_grad()

        with torch.no_grad():
            early_img, late_img, label, early_year, early_month, late_year, late_month = batch
            early_img = early_img.to(self.device)
            late_img = late_img.to(self.device)
            label = label.to(self.device).float()
            early_year = early_year.to(self.device)
            early_month = early_month.to(self.device)
            late_year = late_year.to(self.device)
            late_month = late_month.to(self.device)

        preds = self.net(early_img, late_img, year1=early_year, month1=early_month, year2=late_year, month2=late_month)

        loss = self.net_loss_fn(preds.view(-1), label)

        log_args = dict(sync_dist=False, rank_zero_only=True)

        with torch.no_grad():
            self.log('loss/train', loss, **log_args)

        if self.global_step % 10000 == 0 and self.global_step > 0:
            print("RESETTING TFEVENT FILE")
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()

        return loss

    def validation_step(self, batch, batch_idx):
        self.net.eval()
    
        with torch.no_grad():
            early_img, late_img, label, early_year, early_month, late_year, late_month = batch
            early_img = early_img.to(self.device)
            late_img = late_img.to(self.device)
            label = label.to(self.device).float()
            early_year = early_year.to(self.device)
            early_month = early_month.to(self.device)
            late_year = late_year.to(self.device)
            late_month = late_month.to(self.device)

            preds = self.forward(early_img, late_img, year1=early_year, month1=early_month, year2=late_year, month2=late_month)
            loss = self.net_loss_fn(preds.view(-1), label)
            y_hat = (preds > 0).float()

            log_args = dict(sync_dist=False, rank_zero_only=True)

            self.log('loss/validation', loss, **log_args)

            self.val_acc.update((y_hat, label))
            self.val_recall.update((y_hat, label))
            self.val_precision.update((y_hat, label))

        self.net.train()
        if not self.cfg.finetune:
            if self.cfg.model in ["DINO", "StreetBYOL"]:
                self.net.model.model.eval()
            else:
                self.net.model.eval()

        return {'early_imgs': early_img[:self.cfg.n_imgs].detach().cpu(),
                'late_imgs': late_img[:self.cfg.n_imgs].detach().cpu(),
                'label': label[:self.cfg.n_imgs].detach().cpu(),
                'preds': preds.detach().cpu()}

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        with torch.no_grad():
            val_acc = self.val_acc.compute()
            val_recall = self.val_recall.compute()
            val_precision = self.val_precision.compute()
            self.log("validation/acc", val_acc)
            self.log("validation/recall", val_recall)
            self.log("validation/precision", val_precision)

            if self.trainer.is_global_zero:
                # output_num = 0
                output_num = random.randint(0, len(outputs) -1)
                output = {k: v.detach().cpu() for k, v in outputs[output_num].items()}

                early_imgs = output["early_imgs"]
                late_imgs = output["late_imgs"]
                label = output["label"]
                preds = output["preds"]

                fig, ax = plt.subplots(2, self.cfg.n_imgs, figsize=(self.cfg.n_imgs * 6, 2 * 6))
                for i in range(early_imgs.shape[0]):
                    ax[0, i].imshow(early_imgs[i].permute(1,2,0))
                    ax[0, i].set_title("label: "+str(label[i].item())+" "+"pred: "+str(self.sigmoid(preds[i]).item()), fontsize=10)
                    ax[1, i].imshow(late_imgs[i].permute(1,2,0))
                ax[0, 0].set_ylabel("early_imgs", fontsize=10)
                ax[1, 0].set_ylabel("late_imgs", fontsize=10)
                remove_axes(ax)
                plt.tight_layout()
                add_plot(self.logger.experiment, "TEST_IMAGEs", self.global_step)

            self.val_acc.reset()
            self.val_recall.reset()
            self.val_precision.reset()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs)

    def train_dataloader(self):
        dataset = TestData20KScale(self.cfg, mode="train", time=True)
        return DataLoader(dataset, 
                          self.cfg.batch_size, 
                          shuffle=True, 
                          num_workers=self.cfg.num_workers, 
                          pin_memory=True, 
                          drop_last=True)

    def val_dataloader(self):
        dataset = TestData20KScale(self.cfg, mode="val", time=True)
        return DataLoader(dataset, 
                          self.cfg.val_batch_size, 
                          shuffle=True, 
                          num_workers=self.cfg.num_workers, 
                          pin_memory=True, 
                          drop_last=False)

    def test_dataloader(self):
        dataset = TestData20KScale(self.cfg, mode="test", time=True)
        return DataLoader(dataset, 
                          self.cfg.val_batch_size, 
                          shuffle=True, 
                          num_workers=self.cfg.num_workers, 
                          pin_memory=True, 
                          drop_last=False)

@hydra.main(config_path="configs", config_name="finetune.yml")
def train_main(cfg: DictConfig) -> str:
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

    seed_everything(seed=3407) #3407

    sys.stdout.flush()

    model = Train_SiamModel(cfg)

    tb_logger = TensorBoardLogger(
        name,
        default_hp_metric=False
    )

    if torch.cuda.is_available():
        gpu_args = dict(gpus=-1, accelerator='ddp')
    else:
        gpu_args = dict()

    patience = int(cfg.patience + 50 * (1-cfg.scale))

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        callbacks=[get_early_stop_callback(patience),
                   get_ckpt_callback(checkpoint_dir, name.split("/")[-1]),
                   LearningRateMonitor(logging_interval='step')],
        limit_train_batches=1.0,
        weights_summary=None,
        stochastic_weight_avg=True,
        max_epochs=cfg.max_epochs,
        gradient_clip_val=cfg.gradient_clip_val,
        **gpu_args
    )
    trainer.fit(model)


def test(ckpt_path, gpus=-1, **kwargs):
    task = Train_SiamModel.load_from_checkpoint(ckpt_path)
    if torch.cuda.is_available():
        trainer = Trainer(accelerator="ddp", gpus=gpus)
    else:
        trainer = Trainer()
    trainer.test(task)

def main():
    parser = argparse.ArgumentParser(description="Run training or testing.")
    parser.add_argument("--mode", choices=["train", "test"], help="run mode: train or test")
    parser.add_argument("--checkpoint", required=False, help="path to the checkpoint file")
    args = parser.parse_args()
    prep_args()
    if args.mode == "train":
        print("Running in training mode")
        train_main()
    elif args.mode == "test":
        if args.checkpoint is None:
            parser.error("Test mode requires --checkpoint")
        print(f"Running in testing mode with checkpoint {args.checkpoint}")
        test(args.checkpoint)
    

if __name__ == "__main__":
    main()