"""
Code of *CityPulse: Fine-Grained Assessment of Urban Change with Street View Time Series*
Author: Tianyuan Huang, Zejia Wu, Jiajun Wu, Jackelyn Hwang and Ram Rajagopal

References:

DINOv2: *DINOv2: Learning Robust Visual Features without Supervision*
        https://github.com/facebookresearch/dinov2
        Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, Piotr Bojanowski
"""

import random
import pandas as pd
from tqdm import tqdm
import argparse

import torch
import torch.nn as nn
import torch.multiprocessing
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

from utils import *
from modules import *

import warnings
warnings.filterwarnings("ignore")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class cfg:
    def __init__(self, feature_dim=768, dino_patch_size = 14, model_name="dinov2", model_type="vitb") -> None:
        self.dino_patch_size = dino_patch_size
        self.feature_dim = feature_dim
        self.model_type = model_type
        self.model_name = model_name

class TestData(Dataset):
    def __init__(self, seed=42, time=False, with_label=True, source_idx="\YOUR\DATA\INDEX") -> None:
        super(TestData).__init__()
        random.seed(seed)
        self.time = time
        self.source_idx = source_idx
        self.input_size = 224
        self.with_label = with_label

        self.all_data = pd.read_csv(self.source_idx)

        self.data = self.all_data

        if time:
            self.data['early_month']= self.data['early_path'].apply(lambda x: int(x.split("/")[-1].split("_")[1]))
            self.data['early_year']= self.data['early_path'].apply(lambda x: int(x.split("/")[-1].split("_")[0]))
            self.data['late_month']= self.data['late_path'].apply(lambda x: int(x.split("/")[-1].split("_")[1]))
            self.data['late_year']= self.data['late_path'].apply(lambda x: int(x.split("/")[-1].split("_")[0]))

        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD

        self.transform = [
            transforms.Resize((self.input_size,self.input_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.transform_mean, std=self.transform_std)
        ]
        self.transform = transforms.Compose(self.transform)

        self.img_size = (3, self.input_size, self.input_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        early_path = self.data.loc[ind, "early_path"]
        late_path = self.data.loc[ind, "late_path"]
        if self.with_label:
            label = self.data.loc[ind, "label"]

        early_img = self.transform(Image.open(early_path).convert("RGB"))
        late_img = self.transform(Image.open(late_path).convert("RGB"))

        if self.time:
            early_year = self.data.loc[ind, "early_year"]
            early_month = self.data.loc[ind, "early_month"]
            late_year = self.data.loc[ind, "late_year"]
            late_month = self.data.loc[ind, "late_month"]
            if self.with_label:
                return early_img, late_img, label, early_year, early_month, late_year, late_month
            else:
                return early_img, late_img, early_year, early_month, late_year, late_month
        else:
            if self.with_label:
                return early_img, late_img, label
            else:
                return early_img, late_img

class SiamModel(nn.Module):
    def __init__(self) -> None:
        super(SiamModel, self).__init__()
        self.cfg = cfg()
        self.head = torch.nn.Linear(self.cfg.feature_dim*3, 1, bias=True)
        self.initialize_weights()
        for p in self.head.parameters():
            p.requires_grad = False
        self.head.eval()

        self.model_name = self.cfg.model_name + "_" + self.cfg.model_type + str(self.cfg.dino_patch_size)
        self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        for p in self.model.parameters():
            p.requires_grad = False     
        self.model.eval()
        
    def forward(self, x1, x2):
        if len(x1.shape) == 5:
            x1 = x1.squeeze(1)
        if len(x2.shape) == 5:
            x2 = x2.squeeze(1)
        embed_vec_early = self.model(x1).reshape(x1.shape[0], -1)
        embed_vec_late = self.model(x2).reshape(x2.shape[0], -1)
        x = torch.concat([embed_vec_early, embed_vec_late, embed_vec_late-embed_vec_early], dim=1)
        preds = self.head(x)
        return preds

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)

    def reset_parameters(self):
        with torch.no_grad():
            self.initialize_weights()

class Train_SiamModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = SiamModel()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        return  self.net(x1, x2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Siamese model inference with DINOv2 backbone.")
    parser.add_argument("--data_idx", required=True, help="path to the index of data.")
    parser.add_argument("--checkpoint", required=True, help="path to the finetuned checkpoint file")
    parser.add_argument("--with_label", choices=["True", "False"], help="whether the data are labeled")
    args = parser.parse_args()

    with_label = True if args.with_label == "True" else False
    dataset = TestData(time=False, with_label=with_label, source_idx=args.data_idx)
    loader = DataLoader(dataset, 
                        16, 
                        shuffle=True, 
                        num_workers=8, 
                        pin_memory=True, 
                        drop_last=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Siamese_model = Train_SiamModel.load_from_checkpoint(args.checkpoint).to(device)
    if with_label:
        acc_list = []
        t = tqdm(loader)
        for batch in t:
            x1, x2, label = batch
            pred = Siamese_model(x1.to(Siamese_model.device), x2.to(Siamese_model.device))
            acc = ((Siamese_model.sigmoid(pred)>0.5).type(torch.int8).to("cpu").squeeze(-1)==label).sum()/label.shape[0]
            t.set_description("ACC %f"%acc.item())
            acc_list.append(acc.item())
        print("Accuracy Rate: ", torch.tensor(acc_list).mean().item())
    else:
        y_hat_list = []
        t = tqdm(loader)
        for batch in t:
            x1, x2 = batch
            pred = Siamese_model(x1.to(Siamese_model.device), x2.to(Siamese_model.device))
            y_hat_list = y_hat_list + (Siamese_model.sigmoid(pred)>0.5).type(torch.int8).to("cpu").squeeze(-1).tolist()
        print(y_hat_list)