# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from timm.models.vision_transformer import PatchEmbed, Block
from timm.data import IMAGENET_DEFAULT_MEAN as means
from timm.data import IMAGENET_DEFAULT_STD as stds

from .util.pos_embed import get_2d_sincos_pos_embed

from dataset.transforms import RandomCrop, Pad, RandomApply, Compose, Resize



class RCMaskedMFFIntersectionDMaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with U-VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
        embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
        pre_norm=False, chunks=4, mask_chunks=4, out_indices=[0, 2, 4, 6, 8],
        use_conv=False, random_mask=True, hgds=[], hgds_hyper=[],
        autoweight_loss=False,
        img_mask='mask', feature_mask='mask',
        pretrain=False, **kwargs):
        super().__init__()

        self.pre_norm = pre_norm
        self.pre_normlize = transforms.Compose([
            transforms.Normalize(means, stds),
        ])
        self.pretrain = pretrain

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size*2, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.patch_embed.img_size = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.mask_chunks = mask_chunks
        self.out_indices = out_indices
        proj_layers = [
            torch.nn.Linear(embed_dim, embed_dim)
            for _ in range(len(self.out_indices))
        ]
        self.proj_layers = torch.nn.ModuleList(proj_layers)
        self.proj_weights = torch.nn.Parameter(
            torch.ones(len(self.out_indices)+1).view(-1, 1, 1, 1))

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.chunks = chunks

        self.feature_mask = feature_mask
        self.img_mask = img_mask

        if self.feature_mask == 'noise' or self.img_mask == 'noise':
            betas = torch.linspace(0.0001, 0.02, 1000)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.sqrt_alphas_cumprod = nn.Parameter(
                torch.sqrt(alphas_cumprod), requires_grad=False)
            self.sqrt_one_minus_alphas_cumprod = nn.Parameter(
                torch.sqrt(1. - alphas_cumprod), requires_grad=False)

        self.decoder_head_embed = nn.Linear(
            embed_dim*(self.chunks-1), decoder_embed_dim, bias=True)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_norm = norm_layer(decoder_embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads,
                mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)
        # decoder to patch
        # --------------------------------------------------------------------------

        self.use_conv = use_conv
        self.decoder_rb1 = nn.Conv2d(3, 3, 3, 1, 1)

        self.random_mask = random_mask

        self.norm_pix_loss = norm_pix_loss
        self.pix_loss = nn.L1Loss()

        self.initialize_weights()

        if len(hgds) > 0:
            self.hgds = nn.ModuleList(hgds)
            self.hgds_hyper = hgds_hyper
            self.hgd_loss = nn.CrossEntropyLoss()
            self.hgd_floss = nn.L1Loss()
            num_loss += len(self.hgds_hyper[0])


        self.autoweight_loss = autoweight_loss
        if self.autoweight_loss:
            self.loss_params = nn.Parameter(
                torch.ones(num_loss, requires_grad=True)
            )


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** .5),
            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
            int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02)
        # as cutoff is too big (2.)
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
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def gen_random_masks(self, x, mask_num=2):
        N, C, H, W = x.shape
        p = self.patch_embed.patch_size[0]
        chunks = self.mask_chunks
        L = (H // p)**2

        if self.random_mask:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        else:
            if chunks == 4:
                noise = torch.tensor([[1, 2], [3, 4]])
                noise = noise.repeat_interleave(H // p // 2 + 1, dim=0)
                noise = noise.repeat_interleave(H // p // 2 + 1, dim=1)
            elif chunks == 3:
                noise = torch.tensor([[1], [2], [3]])
                noise = noise.repeat_interleave(H // p // 3 + 1, dim=0)
                noise = noise.repeat_interleave(H // p // 2 + 1, dim=1)
            elif chunks == 2:
                noise = torch.tensor([[1, 2], [2, 1]])
                noise = noise.repeat_interleave(H // p // 2 + 1, dim=0)
                noise = noise.repeat_interleave(H // p // 2 + 1, dim=1)
            # noise = noise.repeat(
            #     int(L ** 0.5 // chunks) + 1, int(L ** 0.5 // chunks) + 1)
            noise = noise[: int(H // p), : int(H // p)]
            noise = torch.unsqueeze(noise, 0)
            noise = noise.repeat(N, 1, 1).to(x.device)
            noise = noise.reshape(N, L)
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        masks = []
        for i in range(mask_num):
            mask = torch.ones([N, L], device=x.device)
            _chunks = torch.randint(2, chunks+1, (1, ))[0]
            ids_index = np.int_(
                np.linspace(0, L - 1, _chunks + 1).round())
            _index = torch.randint(_chunks, (1, ))[0]
            mask[:, ids_index[_index]:ids_index[_index+1]] = 0
            mask = torch.gather(mask, dim=1, index=ids_restore)
            mask = torch.unsqueeze(mask, -1)
            mask = mask.repeat(1, 1, p ** 2 * 3)
            mask = self.unpatchify(mask)
            masks.append(mask)
        return masks


    def random_masking(self, x, ids_restore=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        chunks = self.chunks
        # len_keep = int(L * (1 - mask_ratio))
        assert L % chunks == 0

        if ids_restore is None:
            if self.random_mask:
                noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
            else:
                if chunks == 4:
                    noise = torch.tensor([
                        [1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]])
                elif chunks == 3:
                    noise = torch.tensor([
                        [1, 2, 3], [3, 1, 2], [2, 3, 1]])
                elif chunks == 2:
                    noise = torch.tensor([[1, 2], [2, 1]])
                noise = noise.repeat(
                    int(L ** 0.5 // chunks) + 1, int(L ** 0.5 // chunks) + 1)
                noise = noise[: int(L ** 0.5), : int(L ** 0.5)]
                noise = torch.unsqueeze(noise, 0)
                noise = noise.repeat(N, 1, 1).to(x.device)
                noise = noise.reshape(N, L)

            # sort noise for each sample
            ids_shuffle = torch.argsort(noise, dim=1)
            # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
        else:
            ids_shuffle = torch.argsort(ids_restore, dim=1)

        # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        x_masked = torch.gather(
            x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))
        x_masked = x_masked.reshape(N*chunks, L//chunks, D)

        return x_masked, ids_restore

    def get_index_pos(self, x, pos_embed):
        batches = x.shape[0]
        num_img_patches = x.shape[1]
        D = x.shape[2]
        num_test_img_patches = pos_embed.data.shape[1] - 1
        h_img = int(num_img_patches ** 0.5)
        h_test_img = int(num_test_img_patches ** 0.5)
        x_index = torch.arange(h_img, device=x.device).unsqueeze(0).repeat(h_img, 1)
        x_index = x_index.unsqueeze(0).repeat(batches, 1, 1)
        y_index = torch.transpose_copy(x_index, 1, 2)
        delta = h_test_img - h_img + 1
        if self.training:
            r_x_index = torch.randint(
                delta, size=(batches, 1, 1), device=x.device)
            r_y_index = torch.randint(
                delta, size=(batches, 1, 1), device=x.device)
        else:
            r_x_index = delta // 2 + \
                torch.zeros(size=(batches, 1, 1), dtype=torch.int, device=x.device)
            r_y_index = delta // 2 + \
                torch.zeros(size=(batches, 1, 1), dtype=torch.int, device=x.device)
        index_pos = torch.arange(
            num_test_img_patches, device=x.device).reshape(h_test_img, h_test_img)
        index_pos = index_pos[y_index + r_y_index, x_index + r_x_index]
        index_pos = index_pos.reshape(batches, num_img_patches)
        return index_pos

    def add_pos_embed(self, x, pos_embed, index_pos=None):
        batches = x.shape[0]
        num_img_patches = x.shape[1]
        D = x.shape[2]
        num_test_img_patches = pos_embed.data.shape[1] - 1
        if num_img_patches == num_test_img_patches:
            return x + pos_embed[:, 1:, :]
        if index_pos is None:
            index_pos = self.get_index_pos(x, pos_embed)
        img_pos_embed = torch.gather(
            pos_embed.data[:, 1:, :].repeat(batches, 1, 1),
            dim=1,
            index=index_pos.unsqueeze(-1).repeat(1, 1, D)
        )
        return x + img_pos_embed

    def unnormalize(self, image):
        dtype, device = image.dtype, image.device
        _means = torch.as_tensor(means, dtype=dtype, device=device)
        _stds = torch.as_tensor(stds, dtype=dtype, device=device)
        if _means.ndim == 1:
            _means = _means.view(-1, 1, 1)
        if _stds.ndim == 1:
            _stds = _stds.view(-1, 1, 1)
        return image.mul_(_stds).add_(_means)

    def forward_encoder(self, x):
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        res = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.out_indices:
                proj_x = self.proj_layers[self.out_indices.index(i)](x)
                res.append(proj_x)
        res.append(x)
        res = torch.stack(res)
        proj_weights = F.softmax(self.proj_weights, dim=0)
        res = res * proj_weights
        res = res.sum(dim=0)

        x = self.norm(res)
        return x

    def forward_decoder(self, x, head, ids_restore, index_pos, noise_x):
        N, L, D = x.shape
        chunks = self.chunks
        ids_shuffle = torch.argsort(ids_restore, dim=1)
        # embed tokens -> N*L*D 32*196*1024

        x = self.decoder_embed(x)
        x = self.decoder_embed_norm(x)
        if self.img_mask == 'noise':
            noise_x = self.decoder_embed(noise_x)
            noise_x = self.decoder_embed_norm(noise_x)

        head = head.reshape(N, chunks, D)
        head = head.repeat(chunks, 1, 1)
        _index = torch.ones(chunks, chunks) == 1
        for i in range(chunks):
            _index[i, i] = False
        _index = _index.repeat(1, N).reshape(N * chunks, chunks)
        head = head[_index].reshape(N * chunks, 1, (chunks - 1) * D)
        head = self.decoder_head_embed(head)
        decoder_head_pos_embed = self.decoder_pos_embed.data[:, :1, :].repeat(
            chunks*N, 1, 1)
        head = head + decoder_head_pos_embed

        recon_L = ids_restore.shape[1] // chunks
        x = x.repeat(chunks, 1, 1)
        for i in range(chunks):
            if self.feature_mask == 'noise':
                if self.training:
                    t = torch.randint(200, 1000, size=[N])
                else:
                    t = torch.ones(size=[N], dtype=torch.int) * 600
                if self.img_mask == 'noise':
                    _x = noise_x[:, recon_L * i: recon_L * i + recon_L, :]
                else:
                    _x = x[N * i: N * i + N, recon_L * i: recon_L * i + recon_L, :]
                noise = torch.randn_like(_x)
                _x = self.sqrt_alphas_cumprod[t].view(-1, 1, 1) * _x + \
                     self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1) * noise
                x[N * i: N * i + N, recon_L * i: recon_L * i + recon_L, :] = _x
            elif self.img_mask == 'noise':
                x[N * i: N * i + N, recon_L * i: recon_L * i + recon_L, :] = \
                noise_x[:, recon_L * i: recon_L * i + recon_L, :]
            elif self.feature_mask == 'mask':
                x[N * i: N * i + N, recon_L * i: recon_L * i + recon_L, :] = \
                    self.mask_token
        x = torch.gather(
            x, dim=1,
            index=ids_restore.unsqueeze(-1).repeat(chunks, 1, x.shape[2]))  # unshuffle
        x = self.add_pos_embed(
            x, self.decoder_pos_embed, index_pos.repeat(chunks, 1))
        x = torch.cat([head, x], dim=1)  # append cls token

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            # _x = dsl(torch.cat([x, _skip.pop()], dim=-1))
            x = blk(x)

        # remove cls token
        x = x[:, 1:, :]

        x = self.decoder_norm(x)

        _xs = []
        for i in range(chunks):
            _index = ids_shuffle[:, recon_L * i: recon_L * i + recon_L]
            _index = _index.unsqueeze(-1).repeat(1, 1, x.shape[2])
            _x = torch.gather(x[N * i: N * i + N], dim=1, index=_index)
            _xs.append(_x)
        x = torch.cat(_xs, dim=1)
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        # predictor projection
        x = self.decoder_pred(x)

        if self.use_conv:
            x = self.unpatchify(x)
            x = self.decoder_rb1(x)

        return x

    def forward_loss(self, imgs, pred, clean_imgs=None, labels=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        if not self.use_conv:
            imgs = self.patchify(imgs)
            if clean_imgs is not None:
                clean_imgs = self.patchify(clean_imgs)
        else:
            imgs = imgs
            if clean_imgs is not None:
                clean_imgs = clean_imgs

        if self.norm_pix_loss:
            mean = imgs.mean(dim=-1, keepdim=True)
            var = imgs.var(dim=-1, keepdim=True)
            imgs = (imgs - mean) / (var + 1.e-6) ** .5

        loss_list = []
        if clean_imgs is None:
            pix_loss = self.pix_loss(pred, imgs)
        else:
            pix_loss = self.pix_loss(pred, clean_imgs)
        pix_loss = pix_loss.mean()
        loss_list.append([pix_loss, 'loss', True])
        if clean_imgs is None or len(self.hgds) == 0:
            return loss_list

        pred = self.unnormalize(pred)
        # pixel [0, 1]
        imgs = self.unnormalize(imgs)
        # imgs = imgs * _std + _mean
        if not clean_imgs is None:
            clean_imgs = self.unnormalize(clean_imgs)
            # clean_imgs = clean_imgs * _std + _mean

        hgd_loss = [0] * len(self.hgds_hyper[0])
        total_size = len(pred) * len(self.hgds)
        for hgd, hgd_hyper in zip(self.hgds, self.hgds_hyper):
            hgd.eval()
            with torch.no_grad():
                clean_fea = hgd(clean_imgs)
            if self.mask_chunks == 0:
                _preds = [pred]
            else:
                _preds = []
                mask_num = 1
                masks = self.gen_random_masks(pred, mask_num)
                for mask in masks:
                    _preds.append(imgs * (1 - mask) + pred * mask)
            pred_num = len(_preds)
            recon_fea = hgd(torch.cat(_preds, dim=0))
            for i, fea_key in zip(range(len(hgd_hyper)), recon_fea.keys()):
                _recon_fea = recon_fea[fea_key]
                _clean_fea = clean_fea[fea_key]
                _repeat = [1] * len(_clean_fea.shape)
                _repeat[0] = pred_num
                _hgd_loss = self.hgd_floss(_recon_fea, _clean_fea.repeat(_repeat))
                if self.autoweight_loss:
                    hgd_loss[i] = hgd_loss[i] + _hgd_loss.mean()
                else:
                    hgd_loss[i] = hgd_loss[i] + _hgd_loss.mean() * hgd_hyper[i]
        hgd_loss = [_hgd_loss / len(self.hgds) for _hgd_loss in hgd_loss]
        for i, _hgd_loss in enumerate(hgd_loss):
            loss_list.append([_hgd_loss, 'hgd{}_loss'.format(i), True])
        # loss = loss + pred_loss * 0
        # loss = 0
        return loss_list

    def forward(self, imgs, clean_imgs=None, labels=None):
        if self.pre_norm:
            imgs = self.pre_normlize(imgs)
            if not clean_imgs is None:
                clean_imgs = self.pre_normlize(clean_imgs)

        # embed patches -> N*L*D 32*196*1024
        x = self.patch_embed(imgs)

        # add pos embed w/o cls token
        # x = x + self.pos_embed[:, 1:, :]
        index_pos = self.get_index_pos(x, self.pos_embed)
        x = self.add_pos_embed(x, self.pos_embed, index_pos)
        # masking: length -> length * mask_ratio
        # x, mask, ids_restore = self.random_masking(x, mask_ratio)
        N, L, D = x.shape
        x, ids_restore = self.random_masking(x)
        # -> (N*chunks)*(L/chunks+1)*D 64*99*1024
        latent = self.forward_encoder(x)

        latent_head = latent[:, : 1, :]
        latent = latent[:, 1:, :].reshape(N, L, D)
        noise_latent = None
        if self.img_mask == 'noise':
            if self.training:
                t = torch.randint(200, 600, size=[N])
            else:
                t = torch.ones(size=[N], dtype=torch.int) * 400
            noise = torch.randn_like(imgs)
            noise_imgs = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * imgs + \
                 self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise
            x_noise = self.patch_embed(noise_imgs)
            x_noise = self.add_pos_embed(x_noise, self.pos_embed, index_pos)
            x_noise, _ = self.random_masking(x_noise, ids_restore)
            noise_latent = self.forward_encoder(x_noise)
            noise_latent = noise_latent[:, 1:, :].reshape(N, L, D)

        pred = self.forward_decoder(
            latent, latent_head, ids_restore, index_pos, noise_latent)  # [N, L, p*p*3]

        loss_list = self.forward_loss(imgs, pred, clean_imgs, labels)
        loss = 0
        losses = [l for l in loss_list if l[-1]]
        for i in range(len(losses)):
            loss = loss + losses[i][0]
        show_loss = {}
        for l in loss_list:
            show_loss[l[1]] = l[0]
        return loss, pred, show_loss


def rcmmff_idmae_vit_base_patch16_dec384d8b(**kwargs):
    model = RCMaskedMFFIntersectionDMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def rcmmff_idmae_vit_base_patch16_dec512d8b(**kwargs):
    model = RCMaskedMFFIntersectionDMaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
rcmmff_idmae_vit_base_patch16 = rcmmff_idmae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks







