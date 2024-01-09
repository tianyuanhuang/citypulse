"""
Code of *CityPulse: Fine-Grained Assessment of Urban Change with Street View Time Series*
Author: Tianyuan Huang, Zejia Wu, Jiajun Wu, Jackelyn Hwang and Ram Rajagopal

References:

BYOL: *Bootstrap your own latent: A new approach to self-supervised Learning*
      https://github.com/google-deepmind/deepmind-research/tree/master/byol
      Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, Michal Valko

DINO: *Emerging Properties in Self-Supervised Vision Transformers*
      https://github.com/facebookresearch/dino
      Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin

STEGO: *STEGO: Unsupervised Semantic Segmentation by Distilling Feature Correspondences*
       https://github.com/mhamilton723/STEGO
       Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, William T. Freeman       

Simclr: A Simple Framework for Contrastive Learning of Visual Representations
        https://github.com/google-research/simclr
        Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
"""

import sys
import math
import random
from tqdm import tqdm
from os.path import join

import hydra
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import timm.optim.optim_factory as optim_factory

import torch.multiprocessing
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from utils import *
from modules import *
from data import Data150kBYOL

import warnings
warnings.filterwarnings("ignore")

torch.multiprocessing.set_sharing_strategy('file_system')

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

class BYOLNet(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = cfg
        self.dino_dim = self.cfg.dino_dim
        self.encoder = DinoFeaturizer(self.dino_dim, cfg)
        self.projector = MLP_Projector(self.dino_dim*(int(self.cfg.input_size/self.cfg.dino_patch_size))**2,
                                         self.cfg.mlp_hidden_size, 
                                         self.cfg.projection_size, cfg)
    def forward(self, x):
        x = self.encoder(x)[1]
        x = self.projector(x)
        return x

class BYOLGSVModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.online_network = BYOLNet(cfg)
        self.target_network = BYOLNet(cfg)
        self.initializes_target_network()
        self.predictor = MLP_Projector(self.cfg.projection_size,
                                         self.cfg.mlp_hidden_size, 
                                         self.cfg.projection_size, cfg)
        print(get_parameter_number(self.online_network))

        self.automatic_optimization = False
        self.softmax = nn.Softmax()

        self.m = self.cfg.m

        self.save_hyperparameters()

    def forward(self, x):
        return self.online_network(x)

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

    def adjust_m(self):
        self.m = 1- (1-self.m)*(math.cos(math.pi*self.global_step/self.cfg.max_steps)+1)/2
        return self.m

    def configure_optimizers(self):
        model_params = optim_factory.add_weight_decay(self.online_network, self.cfg.weight_decay)
        model_optim = torch.optim.AdamW(model_params, lr=self.cfg.lr, betas=(0.9, 0.95))

        pre_params = optim_factory.add_weight_decay(self.predictor, self.cfg.weight_decay)
        pre_optim = torch.optim.AdamW(pre_params, lr=self.cfg.lr, betas=(0.9, 0.95))

        return model_optim, pre_optim

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        self.adjust_m()
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def training_step(self, batch, batch_idx):
        model_optim, pre_optim = self.optimizers()
        self.adjust_learning_rate(model_optim)
        self.adjust_learning_rate(pre_optim)
        model_optim.zero_grad()
        pre_optim.zero_grad()

        with torch.no_grad():
            img = batch["img"].to(self.device)
            img_building = batch["img_building"].to(self.device)

        predictions_from_view_1 = self.predictor(self.online_network(img))
        predictions_from_view_2 = self.predictor(self.online_network(img_building))

        with torch.no_grad():
            targets_to_view_2 = self.target_network(img)
            targets_to_view_1 = self.target_network(img_building)
        
        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)

        total_loss = 0
        total_loss += loss.mean()

        log_args = dict(sync_dist=False, rank_zero_only=True)

        with torch.no_grad():
            self.log('loss/total', total_loss, **log_args)

        self.manual_backward(total_loss)
        model_optim.step()
        pre_optim.step()
        self._update_target_network_parameters()

        if self.global_step % 2000 == 0 and self.global_step > 0:
            print("RESETTING TFEVENT FILE")
            # Make a new tfevent file
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()

        return total_loss

    def validation_step(self, batch, batch_idx):
        ind = batch["ind"].to(self.device)
        seq_idx = batch["seq_idx"]
        year = batch["year"].to(self.device) # [BS, seq_len]
        month = batch["month"].to(self.device) # [BS, seq_len]
        img = batch["img"].to(self.device) # [BS, seq_len, 3, 224, 224]
        img_building = batch["img_building"].to(self.device)

        return {'img': img.detach().cpu(),
                'img_buidling': img_building.detach().cpu(),
                "seq_idx": seq_idx,
                "year": year.detach().cpu(),
                "month": month.detach().cpu()}

    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        with torch.no_grad():

            if self.trainer.is_global_zero:
                #output_num = 0
                output_num = random.randint(0, len(outputs) -1)
                output = {k: v.detach().cpu() for k, v in outputs[output_num].items() if k!="seq_idx"}
                output["seq_idx"] = outputs[output_num]["seq_idx"]

                imgs = output["img"]
                img_buildings = output["img_buidling"]
                seq_idxs = output["seq_idx"]
                years = output["year"]
                months = output["month"]

                seq_sample = random.sample(range(0, imgs.shape[0]), self.cfg.n_seq)

                for seq in seq_sample:
                    img = imgs[seq]
                    img_building = img_buildings[seq]
                    seq_idx = seq_idxs[seq]
                    year = years[seq]
                    month = months[seq]
                    n = img.shape[0]

                    with torch.no_grad():
                        fig, ax = plt.subplots(2, n, figsize=(n * 6, 2 * 6))
                        for i in range(n):
                            ax[0, i].set_title(seq_idx+"-"+str(year[i].item())+"-"+str(month[i].item()), fontsize=10)
                            ax[0, i].imshow(img[i].permute(1,2,0))
                            ax[1, i].imshow(img_building[i].cpu().permute(1,2,0))
                        ax[0, 0].set_ylabel("Original_Image", fontsize=10)
                        ax[1, 0].set_ylabel("Buildings", fontsize=10)
                        plt.tight_layout()
                        add_plot(self.logger.experiment, "Seq_IMAGEs", self.global_step)


@hydra.main(config_path="configs", config_name="byol_pretrain.yml")
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

    seed_everything(seed=42)

    sys.stdout.flush()

    train_dataset = Data150kBYOL(cfg, if_val=False)
    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    val_dataset =  Data150kBYOL(cfg, if_val=True)
    val_loader = DataLoader(val_dataset, cfg.eval_batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    model = BYOLGSVModel(cfg)

    tb_logger = TensorBoardLogger(
        name,
        default_hp_metric=False
    )

    if torch.cuda.is_available():
        gpu_args = dict(gpus=-1, accelerator='ddp', val_check_interval=cfg.val_freq)

        if gpu_args["val_check_interval"] > len(train_loader) // 4:
            gpu_args.pop("val_check_interval")
    else:
        gpu_args = dict(accelerator='ddp')

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        max_steps=cfg.max_steps,
        callbacks=[
            ModelCheckpoint(
                dirpath=join(checkpoint_dir, name.split("/")[-1]),
                every_n_train_steps=500,
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