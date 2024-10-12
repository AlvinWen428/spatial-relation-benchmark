import os
import math
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.modules.vision_transformer import VisionTransformer, PatchEmbed, resize_pos_embed, trunc_normal_
from models.modules.read_net import PromptReadoutNet


class RelatiViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_layer=PatchEmbed, 
                 pretrained="", use_attn_mask=True, prompt_emb_pool="max",
                 predicate_dim=30, readnet_d_hidden=512, readnet_dropout=0.0,
                 *args, **kwargs):
        super(RelatiViT, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_layer=embed_layer,
                                        *args, **kwargs)
        self.predicate_dim = predicate_dim

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.cls_token, std=self.init_std)
        assert self.num_tokens == 1, "the number of output tokens should be 1"

        self.use_attn_mask = use_attn_mask
        self.prompt_emb_pool = prompt_emb_pool

        self.n_prompts = (self.patch_embed.grid_size[0] * self.patch_embed.grid_size[1] + 1) * 2
        self.sub_sep_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.obj_sep_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.sub_sep_token, std=self.init_std)
        trunc_normal_(self.obj_sep_token, std=self.init_std)

        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
                torch.zeros(1, self.num_tokens + self.num_patches, self.embed_dim)
            )

        self.load_pretrained(pretrained)

        # subject and object distinguishing embeddings
        self.subj_indicate_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.obj_indicate_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        trunc_normal_(self.subj_indicate_embed, std=self.init_std)
        trunc_normal_(self.obj_indicate_embed, std=self.init_std)

        readout_head_input_dim = self.embed_dim * 2
        self.readout_head = PromptReadoutNet(readout_head_input_dim, readnet_d_hidden, self.predicate_dim, readnet_dropout)

    def load_pretrained(self, pretrained):
        unexpected_keys, missing_keys = [], []
        if pretrained:  # only support load from local file
            checkpoint = torch.load(pretrained, map_location="cpu")
            if 'model' in checkpoint:
                load_state_dict = checkpoint["model"]
            elif 'model_state' in checkpoint:
                load_state_dict = checkpoint["model_state"]
            elif 'state_dict' in checkpoint:
                load_state_dict = checkpoint['state_dict']
            else:
                raise KeyError

            load_state_dict = OrderedDict([(k, v) for k, v in load_state_dict.items() if 'readout_head.' not in k])

            for k in self.state_dict().keys():
                if k not in load_state_dict.keys():
                    missing_keys.append(k)
            for k in load_state_dict.keys():
                if k not in self.state_dict().keys():
                    unexpected_keys.append(k)

            if "pos_embed" in load_state_dict:
                load_state_dict["pos_embed"] = resize_pos_embed(
                    load_state_dict["pos_embed"],
                    self.pos_embed,
                    self.num_tokens,
                    self.patch_embed.grid_size,
                )
            self.load_state_dict(load_state_dict, strict=False)
            print(f"Loading ViT pretrained weights from {pretrained}.")
            print(f"missing keys: {missing_keys}")
            print(f"unexpected keys: {unexpected_keys}")
        else:
            print("Loading ViT pretrained weights from scratch.")

    def get_bbox_image_patches(self, full_im, bbox_s, bbox_o):
        batch_size, _, h, w = full_im.shape
        patch_size = self.patch_embed.patch_size
        sub_masked_img = torch.zeros_like(full_im)
        obj_masked_img = torch.zeros_like(full_im)
        sub_img_mask_slice = []
        obj_img_mask_slice = []
        for i, (sub_box, obj_box) in enumerate(zip(bbox_s, bbox_o)):
            sub_h = min(max(int(sub_box[1] * h) - int(sub_box[0] * h), 4), h - int(sub_box[0] * h))
            sub_w = min(max(int(sub_box[3] * w) - int(sub_box[2] * w), 4), w - int(sub_box[2] * w))
            obj_h = min(max(int(obj_box[1] * h) - int(obj_box[0] * h), 4), h - int(obj_box[0] * h))
            obj_w = min(max(int(obj_box[3] * w) - int(obj_box[2] * w), 4), w - int(obj_box[2] * w))

            sub_masked_img[i, :, int(sub_box[0] * h):int(sub_box[0] * h) + sub_h, int(sub_box[2] * w):int(sub_box[2] * w) + sub_w] \
                = full_im[i, :, int(sub_box[0] * h):int(sub_box[0] * h) + sub_h, int(sub_box[2] * w):int(sub_box[2] * w) + sub_w]
            obj_masked_img[i, :, int(obj_box[0] * h):int(obj_box[0] * h) + obj_h, int(obj_box[2] * w):int(obj_box[2] * w) + obj_w] \
                = full_im[i, :, int(obj_box[0] * h):int(obj_box[0] * h) + obj_h, int(obj_box[2] * w):int(obj_box[2] * w) + obj_w]
            sub_img_mask_slice.append([int(sub_box[0] * h), int(sub_box[0] * h) + sub_h, int(sub_box[2] * w), int(sub_box[2] * w) + sub_w])
            obj_img_mask_slice.append([int(obj_box[0] * h), int(obj_box[0] * h) + obj_h, int(obj_box[2] * w), int(obj_box[2] * w) + obj_w])

        # (B, 3, 224, 224)
        return sub_masked_img, obj_masked_img, torch.tensor(sub_img_mask_slice), torch.tensor(obj_img_mask_slice)

    def prompt_pooling(self, prompt_embedding, img_mask_slice=None):
        if self.prompt_emb_pool == 'max':
            pooled_embedding = torch.max(prompt_embedding, dim=1).values
        elif self.prompt_emb_pool == 'max-in-roi':
            assert img_mask_slice is not None
            rearrange_embedding = rearrange(prompt_embedding, "b (h w) c -> b h w c", h=self.patch_embed.grid_size[0], w=self.patch_embed.grid_size[1])
            pooled_embedding = []
            patch_size_h, patch_size_w = self.patch_embed.patch_size[0], self.patch_embed.patch_size[1]
            for i, (embedding, mask_slice) in enumerate(zip(rearrange_embedding, img_mask_slice)):
                pooled_embedding.append(
                    torch.max(
                        embedding[mask_slice[0]//patch_size_h: math.ceil(mask_slice[1]/patch_size_h),
                        mask_slice[2]//patch_size_w: math.ceil(mask_slice[3]/patch_size_w), :].flatten(0, 1), dim=0
                    ).values
                )
            pooled_embedding = torch.stack(pooled_embedding, dim=0)
        elif self.prompt_emb_pool == 'avg':
            pooled_embedding = torch.mean(prompt_embedding, dim=1)
        elif self.prompt_emb_pool == 'avg-in-roi':
            assert img_mask_slice is not None
            rearrange_embedding = rearrange(prompt_embedding, "b (h w) c -> b h w c", h=self.patch_embed.grid_size[0], w=self.patch_embed.grid_size[1])
            pooled_embedding = []
            patch_size_h, patch_size_w = self.patch_embed.patch_size[0], self.patch_embed.patch_size[1]
            for i, (embedding, mask_slice) in enumerate(zip(rearrange_embedding, img_mask_slice)):
                pooled_embedding.append(
                    torch.mean(
                        embedding[mask_slice[0] // patch_size_h: math.ceil(mask_slice[1] / patch_size_h),
                        mask_slice[2] // patch_size_w: math.ceil(mask_slice[3] / patch_size_w), :].flatten(0, 1), dim=0
                    )
                )
            pooled_embedding = torch.stack(pooled_embedding, dim=0)
        elif self.prompt_emb_pool == 'log-sum-exp':
            pooled_embedding = torch.logsumexp(prompt_embedding, dim=1)
        elif self.prompt_emb_pool == 'logsumexp-in-roi':
            assert img_mask_slice is not None
            rearrange_embedding = rearrange(prompt_embedding, "b (h w) c -> b h w c", h=self.patch_embed.grid_size[0],
                                            w=self.patch_embed.grid_size[1])
            pooled_embedding = []
            patch_size_h, patch_size_w = self.patch_embed.patch_size[0], self.patch_embed.patch_size[1]
            for i, (embedding, mask_slice) in enumerate(zip(rearrange_embedding, img_mask_slice)):
                pooled_embedding.append(
                    torch.logsumexp(
                        embedding[mask_slice[0] // patch_size_h: math.ceil(mask_slice[1] / patch_size_h),
                        mask_slice[2] // patch_size_w: math.ceil(mask_slice[3] / patch_size_w), :].flatten(0, 1), dim=0
                    )
                )
            pooled_embedding = torch.stack(pooled_embedding, dim=0)
        else:
            raise ValueError
        return pooled_embedding

    def forward_features(self, full_im, bbox_s, bbox_o, predicate):
        batch_size = full_im.shape[0]
        input_tensor = full_im

        x = self.patch_embed(input_tensor)  # BNC
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # (B, 3, 224, 224)
        sub_masked_img, obj_masked_img, sub_img_mask_slice, obj_img_mask_slice = self.get_bbox_image_patches(input_tensor, bbox_s, bbox_o)

        sub_prompts = self.patch_embed(sub_masked_img)  # BNC
        obj_prompts = self.patch_embed(obj_masked_img)

        sub_sep_token = self.sub_sep_token.expand(batch_size, -1, -1)  # B,1,C
        obj_sep_token = self.obj_sep_token.expand(batch_size, -1, -1)

        cls_tokens = cls_tokens + self.pos_embed[:, :self.num_tokens, :]
        x = x + self.pos_embed[:, self.num_tokens:, :]
        sub_sep_token = sub_sep_token + self.pos_embed[:, :self.num_tokens, :]
        sub_prompts = sub_prompts + self.pos_embed[:, self.num_tokens:, :] + self.subj_indicate_embed
        obj_sep_token = obj_sep_token + self.pos_embed[:, :self.num_tokens, :]
        obj_prompts = obj_prompts + self.pos_embed[:, self.num_tokens:, :] + self.obj_indicate_embed
        x = torch.cat((cls_tokens, x, sub_sep_token, sub_prompts, obj_sep_token, obj_prompts), dim=1)

        x = self.pos_drop(x)

        # get attention mask
        n_patches = x.shape[1]
        attn_mask = torch.zeros(n_patches, n_patches).to(x.device)
        if self.use_attn_mask:
            attn_mask[self.num_tokens:(self.num_tokens + self.num_patches), (self.num_tokens + self.num_patches):] = -float("inf")
            attn_mask.fill_diagonal_(0)

        for block_idx, block in enumerate(self.blocks):
            x = block(x, attn_mask=attn_mask)
        x = self.norm(x)
        sub_embedding = x[:, self.num_tokens+self.num_patches+1:self.num_tokens+self.num_patches+1+self.num_patches, :]
        obj_embedding = x[:, self.num_tokens+self.num_patches+1+self.num_patches+1:, :]

        sub_embedding = self.prompt_pooling(sub_embedding, sub_img_mask_slice)
        obj_embedding = self.prompt_pooling(obj_embedding, obj_img_mask_slice)

        output = torch.cat((sub_embedding, obj_embedding), dim=-1)
        return output

    def forward_head(self, x, predicate):
        rel_dists = self.readout_head(x)
        predi_onehot = F.one_hot(predicate, num_classes=self.predicate_dim).to(torch.float32)
        return torch.sum(rel_dists * predi_onehot, 1)

    def forward(self, full_im, bbox_s, bbox_o, predicate):
        x = self.forward_features(full_im, bbox_s, bbox_o, predicate)
        output = self.forward_head(x, predicate)
        return output
