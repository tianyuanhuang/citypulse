"""
Code of *CityPulse: Fine-Grained Assessment of Urban Change with Street View Time Series*
Author: Tianyuan Huang, Zejia Wu, Jiajun Wu, Jackelyn Hwang and Ram Rajagopal

References:

MAE: *Masked Autoencoders Are Scalable Vision Learners*
     https://github.com/facebookresearch/mae
     Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Doll{\'a}r and Ross Girshick

DINO: *Emerging Properties in Self-Supervised Vision Transformers*
      https://github.com/facebookresearch/dino
      Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, Armand Joulin

STEGO: *STEGO: Unsupervised Semantic Segmentation by Distilling Feature Correspondences*
       https://github.com/mhamilton723/STEGO
       Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, William T. Freeman       
"""
import torch
import numpy as np
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import PatchEmbed

from utils import *
from attn import Block
import dino.vision_transformer as vits
from pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid_torch

class MLP_Projector(nn.Module):
    def __init__(self, input_dim, middle_dim, output_dim, cfg) -> None:
        super().__init__()
        self.dim = output_dim
        self.middle = middle_dim
        self.cfg = cfg
        self.input_dim = input_dim
        self.layers = torch.nn.Sequential(torch.nn.Linear(self.input_dim, self.middle, bias=True),
                                          nn.BatchNorm1d(self.middle),
                                          torch.nn.ReLU(inplace=True),
                                          torch.nn.Linear(self.middle, self.dim, bias=True))
        self.initialize_weights()

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.layers(x)
        return x

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


class DinoFeaturizer(nn.Module):
    """
    adapted from https://github.com/mhamilton723/STEGO
    """
    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.patch_size = self.cfg.dino_patch_size
        self.feat_type = self.cfg.dino_feat_type
        arch = self.cfg.model_type
        self.model = vits.__dict__[arch](patch_size=self.patch_size, num_classes=0)
        self.dropout = torch.nn.Dropout2d(p=.1)

        if arch == "vit_small" and self.patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and self.patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and self.patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and self.patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        if cfg.pretrained_weights is not None:
            state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
            state_dict = state_dict["teacher"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(cfg.pretrained_weights, msg))
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768

        self.Conv1 = self.make_linear_Conv(self.n_feats)
        self.Conv2 = self.make_nonlinear_Conv(self.n_feats)

        for m in self.Conv1:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        for m in self.Conv2:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def make_linear_Conv(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def make_nonlinear_Conv(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def forward(self, img, n=1, return_class_feat=False):
        
        assert (img.shape[2] % self.patch_size == 0)
        assert (img.shape[3] % self.patch_size == 0)

        # get selected layer activations
        feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
        feat, attn, qkv = feat[0], attn[0], qkv[0]

        feat_h = img.shape[2] // self.patch_size
        feat_w = img.shape[3] // self.patch_size

        if self.feat_type == "feat":
            image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2) # bs, 768, 14, 14
        elif self.feat_type == "KK":
            image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
            B, H, I, J, D = image_k.shape
            image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
        else:
            raise ValueError("Unknown feat type:{}".format(self.feat_type))

        if return_class_feat:
            return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2) # cls

        code = self.Conv1(self.dropout(image_feat))
        code += self.Conv2(self.dropout(image_feat)) # bs, dim, 14, 14

        if self.cfg.dropout:
            return self.dropout(image_feat), code
        else:
            return image_feat, code

class MaskedAutoencoderViT(nn.Module):
    """ 
    Masked Autoencoder with Vision Transformer backbone for Image Sequences
    adapted from https://github.com/facebookresearch/mae
    """
    def __init__(self, seq_len=6, time_embed_dim=128, decoder_time_embed_dim=64, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=True):
        super().__init__()

        self.seq_len = seq_len
        self.time_embed_dim = time_embed_dim
        self.decoder_time_embed_dim = decoder_time_embed_dim

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim-(time_embed_dim*2)), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim-(decoder_time_embed_dim*2)), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def patchify_seq(self, imgs):
        """
        imgs: (N, T, 3, H, W)
        x: (N, T*L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[-2] == imgs.shape[-1] and imgs.shape[-1] % p == 0

        h = w = imgs.shape[-1] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], 3, h, p, w, p))
        x = torch.einsum('ntchpwq->nthwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], imgs.shape[1] * h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def unpatchify_seq(self, x):
        """
        x: (N, T*L, patch_size**2 *3)
        imgs: (N, T, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int((x.shape[1]/self.seq_len)**.5)
        assert h * w * self.seq_len == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], self.seq_len, h, w,  p, p, 3))
        x = torch.einsum('nthwpqc->ntchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], x.shape[1], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D] sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1] [0.2, 0.7, 0.1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove [2, 0, 1]
        ids_restore = torch.argsort(ids_shuffle, dim=1) # [1, 2, 0]

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # index: N,L_keep,D

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def object_masking(self, x, mask_ratio, object_mask=None):
        """
        Generate mask for some specific object in the image, for segmented image case.
        x: [N, L, D], sequence
        """
        if object_mask is None:
            x_masked, mask, ids_restore = self.random_masking(x, mask_ratio)
            return x_masked, mask, ids_restore
        else:
            N, L, D = x.shape  # batch, length, dim
            len_patch = object_mask.sum(dim=-1) # N
            len_keep = (len_patch * (1 - mask_ratio)).int() # N
            len_keep_ulti = int(L * (1 - mask_ratio))
            
            x_masked_list = []
            mask_list = []
            restore_list = []
            for i in range(N):
                nonzero_ids = torch.nonzero(object_mask[i], as_tuple=True)[0] # [len_patch[i],]
                noise = torch.rand(1, nonzero_ids.shape[0], device=x.device) # [1, len_patch[i]]
                ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove [1, len_patch[i]]
                
                bg_ids = torch.tensor(np.setdiff1d(torch.arange(0,L).cpu(), nonzero_ids.cpu())).to(x.device)
                noise_bg = torch.rand(1, bg_ids.shape[0], device=x.device)
                ids_shuffle_bg = torch.argsort(noise_bg, dim=1)

                ids_shuffle_all = torch.cat([nonzero_ids[ids_shuffle][:, :len_keep[i].item()],
                                             bg_ids[ids_shuffle_bg],
                                             nonzero_ids[ids_shuffle][:, len_keep[i].item():]], dim=1) # [1, L]
                
                ids_keep = ids_shuffle_all[:,:len_keep_ulti]
                ids_restore = torch.argsort(ids_shuffle_all, dim=1) # [1, L]
                restore_list.append(ids_restore)

                
                x_masked = torch.gather(x[i].unsqueeze(0), dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D)) # index: [1,len_keep[i],D]
                x_masked_list.append(x_masked)

                mask = torch.ones([1, L], device=x.device)
                mask[:,:-(nonzero_ids.shape[0]-len_keep[i].item())] = 0
                mask = torch.gather(mask, dim=1, index=ids_restore)
                mask_list.append(mask)

            x_masked = torch.cat(x_masked_list, dim=0)
            mask = torch.cat(mask_list, dim=0)
            ids_restore = torch.cat(restore_list, dim=0)

            return x_masked, mask, ids_restore

    def forward_encoder(self, x, year, month, mask_ratio):
        # x : [N, T, C, H, W]
        N, T, C, H, W = x.shape
        self.T = T
        object_mask = self.patchify_seq(x).mean(dim=-1) # (N, L)
        object_mask = (object_mask != 0).float()

        x_list = []
        for i in range(x.shape[1]):
            x_list.append(self.patch_embed(x[:, i]))
        x = torch.cat(x_list, dim=1) # [BS, L*seq_len, D]

        year = year - 2000 # [N, T]
        month = month - 1
        ts_embed = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(self.time_embed_dim, year.reshape(-1).float()),
                                get_1d_sincos_pos_embed_from_grid_torch(self.time_embed_dim, month.reshape(-1).float())], dim=1).float().to(x.device)
        
        ts_cls = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(self.time_embed_dim, torch.tensor([0 for _ in range(x.shape[0])]).float()),
                                get_1d_sincos_pos_embed_from_grid_torch(self.time_embed_dim, torch.tensor([0 for _ in range(x.shape[0])]).float())], dim=1).float().to(x.device)

        
        ts_embed = ts_embed.reshape(-1, T, ts_embed.shape[-1]).unsqueeze(2)
        ts_embed = ts_embed.expand(-1, -1, x.shape[1] // T, -1).reshape(x.shape[0], -1, ts_embed.shape[-1])

        # add pos embed w/o cls token
        x = x + torch.cat([self.pos_embed[:, 1:, :].repeat(ts_embed.shape[0], T, 1), ts_embed], dim=-1)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.object_masking(x, mask_ratio, object_mask=object_mask)

        # append cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) + torch.cat([self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1), ts_cls.unsqueeze(1)], dim=-1)
        # cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, year, month, ids_restore):

        # N = x.shape[0]
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        ts_embed = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(self.decoder_time_embed_dim, year.reshape(-1).float()),
                                get_1d_sincos_pos_embed_from_grid_torch(self.decoder_time_embed_dim, month.reshape(-1).float())], dim=1).float().to(x.device)
        
        ts_cls = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(self.decoder_time_embed_dim, torch.tensor([0 for _ in range(x.shape[0])]).float()),
                                get_1d_sincos_pos_embed_from_grid_torch(self.decoder_time_embed_dim, torch.tensor([0 for _ in range(x.shape[0])]).float())], dim=1).float().to(x.device)
        
        ts_embed = ts_embed.reshape(-1, self.T, ts_embed.shape[-1]).unsqueeze(2)
        ts_embed = ts_embed.expand(-1, -1, x.shape[1] // self.T, -1).reshape(x.shape[0], -1, ts_embed.shape[-1])

        ts_embed = torch.cat([ts_cls.unsqueeze(1), ts_embed], dim=1)
        pos_embed = torch.cat([self.decoder_pos_embed[:, :1, :], 
                               self.decoder_pos_embed[:, 1:, :].repeat(1, self.T, 1)], dim=1).expand(ts_embed.shape[0], -1, -1)
        x = x + torch.cat([pos_embed, ts_embed], dim=-1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, T, 3, H, W]
        pred: [N, T*L, p*p*3]
        mask: [N, T*L], 0 is keep, 1 is remove, 
        """
        target = self.patchify_seq(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, T*L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, year, month, mask_ratio=0.5):
        latent, mask, ids_restore = self.forward_encoder(imgs, year, month, mask_ratio)
        pred = self.forward_decoder(latent, year, month, ids_restore) # [N, T*L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask

    def visualization(self, seq_index, imgs, year, month, mask_ratio=0.5, path="./"):
        T = imgs.shape[1]
        with torch.no_grad():
            latent, mask, ids_restore = self.forward_encoder(imgs, year, month, mask_ratio)
            preds = self.forward_decoder(latent, year, month, ids_restore) # [N, T*L, p*p*3]

            bs = imgs.shape[0]
            idx = np.random.choice(range(0,bs))
            year = year[idx]
            month = month[idx]
            img = self.patchify_seq(imgs[idx].unsqueeze(0))
            img[:, mask[idx].bool(), :] = 0.5
            img = self.unpatchify_seq(img) # [1, T, 3, 224, 224]
            
            pred = preds[idx].unsqueeze(0)
            pred[:, (1-mask[idx]).bool(), :] = 0
            img_pred = self.patchify_seq(imgs[idx].unsqueeze(0))
            img_pred[:, mask[idx].bool(), :] = 0
            pred = pred + img_pred
            pred = self.unpatchify_seq(pred) # [1, T, 3, 224, 224]

            ori_img = imgs[idx].unsqueeze(0)

            fig, ax = plt.subplots(3, T, figsize=(T * 6, 3 * 6), dpi=200)
            for i in range(T):
                ax[0, i].set_title(seq_index+"-"+str(year[i].item())+"-"+str(month[i].item()), fontsize=18)
                ax[0, i].imshow(ori_img[0, i].permute(1,2,0))
                ax[1, i].imshow(img[0, i].permute(1,2,0))
                ax[2, i].imshow(pred[0, i].permute(1,2,0))
            ax[0, 0].set_ylabel("Original Image", fontsize=18)
            ax[1, 0].set_ylabel("Masked Image", fontsize=18)
            ax[2, 0].set_ylabel("Reconstructed Image", fontsize=18)
            remove_axes(ax)
            plt.savefig(os.path.join(path, seq_index+".png"))


def mae_vit(cfg, **kwargs):
    model = MaskedAutoencoderViT(
        seq_len=cfg.seq_len, time_embed_dim=cfg.time_embed_dim, img_size=cfg.img_size,
        in_chans = cfg.in_chans, patch_size=cfg.patch_size, embed_dim=cfg.embed_dim, depth=cfg.depth, num_heads=cfg.num_heads,
        decoder_embed_dim=cfg.decoder_embed_dim, decoder_depth=cfg.decoder_depth, decoder_num_heads=cfg.decoder_num_heads,
        mlp_ratio=cfg.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=True, **kwargs)
    return model