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

STEGO: *STEGO: Unsupervised Semantic Segmentation by Distilling Feature Correspondences*
       https://github.com/mhamilton723/STEGO
       Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, William T. Freeman

Simclr: A Simple Framework for Contrastive Learning of Visual Representations
        https://github.com/google-research/simclr
        Ting Chen, Simon Kornblith, Mohammad Norouzi, Geoffrey Hinton
"""


import os
import math
import random
from tqdm import tqdm
from PIL import Image

import torch
import numpy as np
import pandas as pd
import torch.multiprocessing
from torch.utils.data import Dataset
from torchvision import transforms

import clip
from gaussian_blur import GaussianBlur


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_simclr_data_transforms(input_shape, s=1):
    # get a set of data augmentation transformations as described in the SimCLR paper.
    color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    data_transforms = transforms.Compose([transforms.Resize((input_shape[0],input_shape[1]), interpolation=Image.BICUBIC),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomGrayscale(p=0.2),
                                          GaussianBlur(kernel_size=int(0.1 * input_shape[0])),
                                          transforms.ToTensor()])
    return data_transforms

class TestData20KScale(Dataset):
    def __init__(self, cfg, seed=3407, mode="train", time=False, source_idx=None) -> None:
        super(TestData20KScale).__init__()
        random.seed(seed)
        self.cfg = cfg
        self.scale = self.cfg.scale
        self.time = time
        self.source_idx = self.cfg.source_idx if source_idx==None else source_idx
        self.input_size = self.cfg.input_size

        self.all_data = pd.read_csv(self.source_idx)
        self.all_seq = self.all_data["seq_index"].unique()
        self.len_seq = len(self.all_seq)

        np.random.seed(seed)
        self.test_seq = np.random.choice(self.all_seq, size=int(self.len_seq*0.5), replace=False)
        self.all_seq = np.sort(np.array(list(set(self.all_seq)-set(self.test_seq))))

        np.random.seed(seed)
        self.all_seq = np.random.choice(self.all_seq, size=int(len(self.all_seq)*self.scale), replace=False)
        self.len_seq = len(self.all_seq)

        val_split_point = int(self.len_seq * 0.1)

        if val_split_point <= 1:
            self.train_seq = self.all_seq[:-1]
            self.val_seq = self.all_seq[-1:]
        else:
            self.val_seq = self.all_seq[-val_split_point:]
            self.train_seq = self.all_seq[:-val_split_point]

        if mode == "train":
            self.len_seq = len(self.train_seq)
            self.data = self.all_data[self.all_data["seq_index"].isin(self.train_seq)].reset_index(drop=True)
        elif mode == "val":
            self.len_seq = len(self.val_seq)
            self.data = self.all_data[self.all_data["seq_index"].isin(self.val_seq)].reset_index(drop=True)
        elif mode == "test":
            self.len_seq = len(self.test_seq)
            self.data = self.all_data[self.all_data["seq_index"].isin(self.test_seq)].reset_index(drop=True)
        else:
            raise

        if time:
            self.data['early_month']= self.data['early_path'].apply(lambda x: int(x.split("/")[-1].split("_")[1]))
            self.data['early_year']= self.data['early_path'].apply(lambda x: int(x.split("/")[-1].split("_")[0]))
            self.data['late_month']= self.data['late_path'].apply(lambda x: int(x.split("/")[-1].split("_")[1]))
            self.data['late_year']= self.data['late_path'].apply(lambda x: int(x.split("/")[-1].split("_")[0]))

        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.img_size = (3, self.input_size, self.input_size)
        
        if self.cfg.model == "CLIP":
            model_name = self.cfg.model_type + "/" + str(self.cfg.dino_patch_size)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.transform = clip.load(model_name, device)[1]
        
        else:
            self.transform = [
                transforms.Resize((self.input_size,self.input_size), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.transform_mean, std=self.transform_std)
            ]
            self.transform = transforms.Compose(self.transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ind):
        early_path = self.data.loc[ind, "early_path"]
        late_path = self.data.loc[ind, "late_path"]
        label = self.data.loc[ind, "label"]

        early_img = self.transform(Image.open(early_path).convert("RGB"))
        late_img = self.transform(Image.open(late_path).convert("RGB"))

        if self.time:
            early_year = self.data.loc[ind, "early_year"]
            early_month = self.data.loc[ind, "early_month"]
            late_year = self.data.loc[ind, "late_year"]
            late_month = self.data.loc[ind, "late_month"]
            return early_img, late_img, label, early_year, early_month, late_year, late_month
        else:
            return early_img, late_img, label


class Data150kBYOL(Dataset):
    def __init__(self, cfg, if_val=False, source_idx=None) -> None:
        super(Data150kBYOL).__init__()
        self.cfg = cfg
        self.source_idx = self.cfg.source_idx if source_idx==None else source_idx
        self.data_root = self.cfg.data_root

        self.input_size = self.cfg.input_size
        self.drop_rate = self.cfg.drop_rate
        self.seq_len = self.cfg.seq_len

        self.img_root = os.path.join(self.data_root, "img")
        self.img_building_root = os.path.join(self.data_root, "img_building")

        self.all_imgs = os.listdir(self.img_root)

        self.idx_df = pd.read_csv(self.source_idx)
        self.idx_df["flag"] = (self.idx_df["building_per"] > self.drop_rate).apply(lambda x: int(x))
        self.idx_df = self.idx_df.loc[self.idx_df["flag"]==1,:].reset_index(drop=True)
        
        self.if_val = if_val
        if self.if_val:
            self.delete_seq()
        self.all_idx = np.sort(self.idx_df["idx"].unique())

        self.img_size = (3, self.input_size, self.input_size)

        self.transform = get_simclr_data_transforms(self.img_size[::-1], s=self.cfg.s)

        self.seq_list = self.idx_df["seq_idx"].unique()

        if self.if_val:
            self.seq_list = list(self.seq_list[-50:]) + list(self.seq_list[:50])

    def delete_seq(self):
        self.idx_df["flag"] = 1
        del_array = self.idx_df.groupby("seq_idx").sum()["flag"][
            self.idx_df.groupby("seq_idx").sum()["flag"]<self.seq_len].index.values
        del_index_list = []
        for idx in range(len(self.idx_df)):
            if self.idx_df.iloc[idx,:]["seq_idx"] in del_array:
                del_index_list.append(idx)
        print(f">>> Drop {len(del_array)} sequences, which is {len(del_index_list)} images!")
        # print(del_array)
        self.idx_df = self.idx_df.drop(index=del_index_list, axis=1).reset_index(drop=True)

    def __len__(self):
        if self.if_val:
            return len(self.seq_list)
        else:
            return len(self.all_idx)

    def __getitem__(self, ind):
        if not self.if_val:
            idx = self.idx_df.iloc[ind, 0]
            year = self.idx_df.iloc[ind, 2]
            month = self.idx_df.iloc[ind, 3]

            img_building = self.transform(Image.open(os.path.join(self.img_building_root, "img_building_"+str(idx)+".jpg")).convert("RGB"))
            img = self.transform(Image.open(os.path.join(self.img_root, "img_"+str(idx)+".jpg")).convert("RGB"))

            ret = {"ind": ind,
                "idx": idx,
                "img": img,
                "img_building": img_building,
                "year": year,
                "month": month,
                }
            return ret
        else:
            seq_idx = self.seq_list[ind]
            query_seq = self.idx_df.loc[self.idx_df["seq_idx"]==seq_idx]
            query_seq = query_seq.loc[np.sort(np.random.choice(query_seq.index, (self.cfg.seq_len,), replace=False))].reset_index(drop=True)
            all_idx = query_seq["idx"].values
            year = query_seq["year"].values
            month = query_seq["month"].values

            img_building_list = []
            img_list = []

            for idx in all_idx:
                img_building = self.transform(Image.open(os.path.join(self.img_building_root, "img_building_"+str(idx)+".jpg")).convert("RGB"))
                img_building_list.append(img_building)

                img = self.transform(Image.open(os.path.join(self.img_root, "img_"+str(idx)+".jpg")).convert("RGB"))
                img_list.append(img)

            ret = {"ind": ind,
                "seq_idx": seq_idx,
                "year": year,
                "month": month,
                "img_building": torch.stack(img_building_list, dim=0),
                "img": torch.stack(img_list, dim=0),
                }
            return ret

class Data150kIMG(Dataset):
    def __init__(self, cfg, building_img_only=True, img_only=False, if_seq=True, if_val=False, with_pos=False, source_idx=None) -> None:
        '''
        building_img_only: if this is set to be true, only segmented building part of the image will be loaded
        img_only: if this is set to be true, only original image will be loaded, can not be true when ``building_img_only`` is true
        if_seq: batches of image sequences will be loaded instead of batches of images if set to be true
        if_val: for validation, only a small proportion of dataset will be loaded if set to be true
        with_pos: if set to be true, the corresponding positive samples will also be loaded 
        '''
        super(Data150kIMG).__init__()
        self.cfg = cfg
        self.source_idx = self.cfg.source_idx if source_idx==None else source_idx
        self.data_root = r"/YOUR/DATA/ROOT"
        self.building_img_only = building_img_only
        self.img_only = img_only
        if self.building_img_only and self.img_only:
            raise
        self.with_pos = with_pos
        self.input_size = self.cfg.input_size
        self.drop_rate = self.cfg.drop_rate
        self.seq_len = self.cfg.seq_len

        self.img_root = os.path.join(self.data_root, "img")
        self.img_bg_root = os.path.join(self.data_root, "img_bg")
        self.img_building_root = os.path.join(self.data_root, "img_building")
        self.pos_img_root = os.path.join(self.data_root, "pos_img")
        self.pos_building_root = os.path.join(self.data_root, "pos_building")
        self.pos_bg_root = os.path.join(self.data_root, "pos_bg")

        self.all_imgs = os.listdir(self.img_root)
        self.all_idx = sorted([int(_.split("_")[-1].split(".")[0]) for _ in self.all_imgs])

        self.idx_df = pd.read_csv(r"/YOUR/INDEX/FILE.csv") 
        # this csv file contains five columns: idx (the index of the image in the dataset), seq_index (the sequence index to which the image belongs), year, month, building_per (the proportion of building pixels in the picture)
        self.idx_df["flag"] = (self.idx_df["building_per"] > self.drop_rate).apply(lambda x: int(x))
        print(">>> Drop Rate: ", 1-self.idx_df["flag"].mean())
        self.idx_df = self.idx_df.loc[self.idx_df["flag"]==1,:].reset_index(drop=True)
        self.delete_seq()
        self.all_idx = np.sort(self.idx_df["idx"].unique())

        self.transform = [
            transforms.Resize((self.input_size,self.input_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ]
        self.transform = transforms.Compose(self.transform)

        self.img_size = (3, self.input_size, self.input_size)
        
        self.idx_seq = np.load(r"/Image/Index/in/Every/Sequence.npy", allow_pickle=True) # The index of all the images in each sequence is stored in this npy file.
        self.seq_list = self.idx_df["seq_idx"].unique()
        self.if_seq = if_seq
        self.if_val = if_val

        if self.if_val and (not self.if_seq):
            raise

        if self.if_val:
            self.seq_list = list(self.seq_list[-50:]) + list(self.seq_list[:50])

    def delete_seq(self):
        self.idx_df["flag"] = 1
        del_array = self.idx_df.groupby("seq_idx").sum()["flag"][
            self.idx_df.groupby("seq_idx").sum()["flag"]<self.seq_len].index.values
        del_index_list = []
        for idx in range(len(self.idx_df)):
            if self.idx_df.iloc[idx,:]["seq_idx"] in del_array:
                del_index_list.append(idx)
        print(f">>> Drop {len(del_array)} sequences, which is {len(del_index_list)} images!")
        self.idx_df = self.idx_df.drop(index=del_index_list, axis=1).reset_index(drop=True)

    def __len__(self):
        if self.if_seq:
            return len(self.seq_list)
        else:
            return len(self.all_idx)

    def __getitem__(self, ind):
        if not self.if_seq:
            idx = self.idx_df.iloc[ind, 0]
            year = self.idx_df.iloc[ind, 2]
            month = self.idx_df.iloc[ind, 3]

            img_building = self.transform(Image.open(os.path.join(self.img_building_root, "img_building_"+str(idx)+".jpg")).convert("RGB"))
            img = self.transform(Image.open(os.path.join(self.img_root, "img_"+str(idx)+".jpg")).convert("RGB"))

            if self.building_img_only:
                if self.with_pos:
                    pos_building = self.transform(Image.open(os.path.join(self.pos_building_root, "pos_building_"+str(idx)+".jpg")).convert("RGB"))
                    ret = {"ind": ind,
                       "idx": idx,
                       "img_building": img_building,
                       "pos_building": pos_building,
                       "year": year,
                       "month": month,
                       }
                    return ret
                else:
                    ret = {"ind": ind,
                        "idx": idx,
                        "img_building": img_building,
                        "year": year,
                        "month": month,
                        }
                    return ret
            elif self.img_only:
                if self.with_pos:
                    pos_img = self.transform(Image.open(os.path.join(self.pos_img_root, "pos_img_"+str(idx)+".jpg")).convert("RGB"))
                    ret = {"ind": ind,
                       "idx": idx,
                       "img_building": img,
                       "pos_building": pos_img,
                       "year": year,
                       "month": month,
                       }
                    return ret
                else:
                    ret = {"ind": ind,
                        "idx": idx,
                        "img_building": img,
                        "year": year,
                        "month": month,
                        }
                    return ret
            else:
                pos_building = self.transform(Image.open(os.path.join(self.pos_building_root, "pos_building_"+str(idx)+".jpg")).convert("RGB"))
                img_bg = self.transform(Image.open(os.path.join(self.img_bg_root, "img_bg_"+str(idx)+".jpg")).convert("RGB"))
                pos_img = self.transform(Image.open(os.path.join(self.pos_img_root, "pos_img_"+str(idx)+".jpg")).convert("RGB"))
                pos_bg = self.transform(Image.open(os.path.join(self.pos_bg_root, "pos_bg_"+str(idx)+".jpg")).convert("RGB"))

                all_seq_idx = self.idx_seq[idx]
                rdm_imgs = np.sort(np.random.choice(list(range(len(all_seq_idx))), 3, replace=False))
                img_chosen = all_seq_idx[rdm_imgs]

                img_chosen_list = []
                for img_idx in img_chosen:
                    img_chosen_list.append(self.transform(Image.open(os.path.join(self.img_root, "img_"+str(img_idx)+".jpg")).convert("RGB")))
                    # assert img_chosen_list[-1].shape == self.img_size
                img_chosen_label = np.random.choice([0,1], 1, replace=False)[0]

                ret = {"ind": ind,
                       "idx": idx,
                       "img": img,
                       "img_building": img_building,
                       "img_bg": img_bg,
                       "pos_img": pos_img,
                       "pos_building": pos_building,
                       "pos_bg": pos_bg,
                       "img_chosen": torch.stack(img_chosen_list, dim=0),
                       "img_chosen_label": img_chosen_label,
                       "year": year,
                       "month": month,
                    }
                return ret
        else:
            seq_idx = self.seq_list[ind]
            query_seq = self.idx_df.loc[self.idx_df["seq_idx"]==seq_idx]
            query_seq = query_seq.loc[np.sort(np.random.choice(query_seq.index, (self.cfg.seq_len,), replace=False))].reset_index(drop=True)
            all_idx = query_seq["idx"].values
            year = query_seq["year"].values
            month = query_seq["month"].values

            img_building_list = []
            img_list = []

            for idx in all_idx:
                img_building = self.transform(Image.open(os.path.join(self.img_building_root, "img_building_"+str(idx)+".jpg")).convert("RGB"))
                img_building_list.append(img_building)

                img = self.transform(Image.open(os.path.join(self.img_root, "img_"+str(idx)+".jpg")).convert("RGB"))
                img_list.append(img)

            if self.building_img_only:
                if self.with_pos:
                    pos_building_list = []
                    for idx in all_idx:
                        pos_building = self.transform(Image.open(os.path.join(self.pos_building_root, "pos_building_"+str(idx)+".jpg")).convert("RGB"))
                        pos_building_list.append(pos_building)
                    ret = {"ind": ind,
                       "seq_idx": seq_idx,
                       "year": year,
                       "month": month,
                       "img_building": torch.stack(img_building_list, dim=0),
                       "pos_building": torch.stack(pos_building_list, dim=0),
                       }
                    return ret
                else:
                    ret = {"ind": ind,
                        "seq_idx": seq_idx,
                        "year": year,
                        "month": month,
                        "img_building": torch.stack(img_building_list, dim=0),
                        }
                    return ret

            elif self.img_only:
                if self.with_pos:
                    pos_img_list = []
                    for idx in all_idx:
                        pos_img = self.transform(Image.open(os.path.join(self.pos_img_root, "pos_img_"+str(idx)+".jpg")).convert("RGB"))
                        pos_img_list.append(pos_img)
                    ret = {"ind": ind,
                       "seq_idx": seq_idx,
                       "year": year,
                       "month": month,
                       "img_building": torch.stack(img_list, dim=0),
                       "pos_buidling": torch.stack(pos_img_list, dim=0),
                       }
                    return ret
                else:
                    ret = {"ind": ind,
                        "seq_idx": seq_idx,
                        "year": year,
                        "month": month,
                        "img_building": torch.stack(img_list, dim=0),
                        }
                    return ret

            else:
                img_bg_list = []
                pos_img_list = []
                pos_bg_list = []
                pos_building_list = []

                for idx in all_idx:
                    pos_building = self.transform(Image.open(os.path.join(self.pos_building_root, "pos_building_"+str(idx)+".jpg")).convert("RGB"))
                    pos_building_list.append(pos_building)

                    img_bg = self.transform(Image.open(os.path.join(self.img_bg_root, "img_bg_"+str(idx)+".jpg")).convert("RGB"))
                    img_bg_list.append(img_bg)

                    pos_img = self.transform(Image.open(os.path.join(self.pos_img_root, "pos_img_"+str(idx)+".jpg")).convert("RGB"))
                    pos_img_list.append(pos_img)

                    pos_bg = self.transform(Image.open(os.path.join(self.pos_bg_root, "pos_bg_"+str(idx)+".jpg")).convert("RGB"))
                    pos_bg_list.append(pos_bg)

                ret = {"ind": ind,
                       "seq_idx": seq_idx,
                       "year": year,
                       "month": month,
                       "img": torch.stack(img_list, dim=0),
                       "img_building": torch.stack(img_building_list, dim=0),
                       "img_bg": torch.stack(img_bg_list, dim=0),
                       "pos_img": torch.stack(pos_img_list, dim=0),
                       "pos_building": torch.stack(pos_building_list, dim=0),
                       "pos_bg": torch.stack(pos_bg_list, dim=0)
                       }
                return ret