import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import numpy as np
from einops import rearrange
import copy

from models.modules.vision_transformer import VisionTransformer, Attention, Block, resize_pos_embed, trunc_normal_
from models.modules.read_net import PromptReadoutNet


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0.0, proj_drop=0.0):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key, value, attn_mask=None):
        B, N_q, C = query.shape
        q = self.proj_q(query)  # (B, N_q, dim)
        k = self.proj_k(key)  # (B, N_k, dim)
        v = self.proj_v(value)  # (B, N_v, dim)

        # (B, N, dim) --> (B, N, num_heads, dim // num_heads) --> (B, num_heads, N, dim // num_heads)
        q = q.reshape(*q.shape[:-1], self.num_heads, q.shape[-1] // self.num_heads).permute(0, 2, 1, 3)
        k = k.reshape(*k.shape[:-1], self.num_heads, k.shape[-1] // self.num_heads).permute(0, 2, 1, 3)
        v = v.reshape(*v.shape[:-1], self.num_heads, v.shape[-1] // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N_q, N_k)
        if attn_mask is not None:
            attn += attn_mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # (B, num_heads, N_q, dim // num_heads) --> (B, N_q, num_heads, dim // num_heads) --> (B, N_q, dim)
        x = (attn @ v).transpose(1, 2).reshape(B, N_q, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DecoderBlock(Block):
    def __init__(self, dim, num_heads, norm_layer=nn.LayerNorm, drop=0.0, attn_drop=0.0, *args, **kwargs):
        super(DecoderBlock, self).__init__(dim=dim, num_heads=num_heads, drop=drop, attn_drop=attn_drop,
                                           *args, **kwargs)
        self.norm0 = norm_layer(dim)
        self.attn = None
        self.self_attn = Attention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

    def forward(self, x, ctx_tensor, attn_mask=None):
        x_self_attn_output = self.self_attn(self.norm0(x), attn_mask=attn_mask)

        if self.init_values is None:
            x = x + self.drop_path(x_self_attn_output)
        else:
            x = x + self.drop_path(self.gamma_1 * x_self_attn_output)

        x_cross_attn_output = self.cross_attn(query=self.norm1(x), key=ctx_tensor, value=ctx_tensor, attn_mask=attn_mask)

        if self.init_values is None:
            x = x + self.drop_path(x_cross_attn_output)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * x_cross_attn_output)
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class VitDecoder(nn.Module):
    def __init__(self, depth=1, embed_dim=768, num_heads=12, drop_rate=0.0, attn_drop_rate=0.0,
                 norm_layer=None, act_layer=None):
        super(VitDecoder, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                    act_layer=act_layer
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

    def forward(self, x, ctx, attn_mask=None):
        for i, block in enumerate(self.blocks):
            x = block(x, ctx, attn_mask)
        x = self.norm(x)
        return x


class CrossAttnViT(VisionTransformer):
    def __init__(self,
                 embed_dim=768,
                 pretrained="",
                 num_heads=12,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 decoder_depth=1,
                 decoder_drop_rate=0.0,
                 decoder_attn_drop_rate=0.0,
                 readnet_d_hidden=512,
                 readnet_dropout=0.0,
                 predicate_dim=30,
                 emb_pool='max-in-roi',
                 *args,
                 **kwargs,):
        super(CrossAttnViT, self).__init__(embed_dim=embed_dim, num_heads=num_heads, drop_rate=drop_rate,
                                           attn_drop_rate=attn_drop_rate, *args, **kwargs)

        self.head = None
        self.pre_logits = None
        assert self.num_tokens == 1, "the number of output tokens should be 1"

        self.load_pretrained(pretrained)

        self.predicate_dim = predicate_dim
        self.num_queries = 2

        self.decoder = VitDecoder(depth=decoder_depth, embed_dim=embed_dim, num_heads=num_heads, drop_rate=decoder_drop_rate,
                                  attn_drop_rate=decoder_attn_drop_rate)

        self.pos_word_embed = nn.Linear(9 + 300, self.embed_dim)
        self.query_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_queries, embed_dim)
        )
        trunc_normal_(self.query_pos_embed, std=self.init_std)

        self.head = PromptReadoutNet(self.embed_dim * self.num_queries, readnet_d_hidden, self.predicate_dim, readnet_dropout)
        
        self.emb_pool = emb_pool
        self.num_patches = self.patch_embed.num_patches
        self.sub_indicate_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.obj_indicate_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.sub_indicate_embed, std=self.init_std)
        trunc_normal_(self.obj_indicate_embed, std=self.init_std)

    def load_pretrained(self, pretrained):
        unexpected_keys, missing_keys = [], []
        if pretrained:  # only support load from local file
            checkpoint = torch.load(pretrained, map_location="cpu")
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            for k in self.state_dict().keys():
                if k not in state_dict.keys():
                    missing_keys.append(k)
            for k in state_dict.keys():
                if k not in self.state_dict().keys():
                    unexpected_keys.append(k)

            if "pos_embed" in state_dict:
                state_dict["pos_embed"] = resize_pos_embed(
                    state_dict["pos_embed"],
                    self.pos_embed,
                    self.num_tokens,
                    self.patch_embed.grid_size,
                )
            self.load_state_dict(state_dict, strict=False)
            print(f"Loading ViT pretrained weights from {pretrained}.")
            print(f"missing keys: {missing_keys}")
            print(f"unexpected keys: {unexpected_keys}")
        else:
            print("Loading ViT pretrained weights from scratch.")

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert (
                self.embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature ** omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        assert self.num_tokens == 1, "Assuming one and only one token, [cls]"
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False

    def encode(self, input_tensor):
        x = self.patch_embed(input_tensor)  # BNC
        x = x + self.pos_embed[:, self.num_tokens:, :]
        x = self.pos_drop(x)

        x = self.blocks(x)
        x = self.norm(x)
        return x

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
            # sub_masked_img[i, :, :sub_h, :sub_w] \
            #     = full_im[i, :, int(sub_box[0] * h):int(sub_box[0] * h) + sub_h, int(sub_box[2] * w):int(sub_box[2] * w) + sub_w]
            # obj_masked_img[i, :, :obj_h, :obj_w] \
            #     = full_im[i, :, int(obj_box[0] * h):int(obj_box[0] * h) + obj_h, int(obj_box[2] * w):int(obj_box[2] * w) + obj_w]
            sub_img_mask_slice.append([int(sub_box[0] * h), int(sub_box[0] * h) + sub_h, int(sub_box[2] * w), int(sub_box[2] * w) + sub_w])
            obj_img_mask_slice.append([int(obj_box[0] * h), int(obj_box[0] * h) + obj_h, int(obj_box[2] * w), int(obj_box[2] * w) + obj_w])

        # (B, 3, 224, 224)
        return sub_masked_img, obj_masked_img, torch.tensor(sub_img_mask_slice), torch.tensor(obj_img_mask_slice)

    def feature_pooling(self, embedding, img_mask_slice=None):
        if self.emb_pool == 'max':
            pooled_embedding = torch.max(embedding, dim=1).values
        elif self.emb_pool == 'max-in-roi':
            assert img_mask_slice is not None
            rearrange_embedding = rearrange(embedding, "b (h w) c -> b h w c", h=self.patch_embed.grid_size[0], w=self.patch_embed.grid_size[1])
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
        elif self.emb_pool == 'avg':
            pooled_embedding = torch.mean(embedding, dim=1)
        elif self.emb_pool == 'avg-in-roi':
            assert img_mask_slice is not None
            rearrange_embedding = rearrange(embedding, "b (h w) c -> b h w c", h=self.patch_embed.grid_size[0], w=self.patch_embed.grid_size[1])
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
        elif self.emb_pool == 'log-sum-exp':
            pooled_embedding = torch.logsumexp(embedding, dim=1)
        else:
            raise ValueError
        return pooled_embedding

    def forward_features(self, full_im, bbox_s, bbox_o, predicate):
        input_tensor = full_im

        # encode the image
        image_ctx_tokens = self.encode(input_tensor)

        # (B, 3, 224, 224)
        sub_masked_img, obj_masked_img, sub_img_mask_slice, obj_img_mask_slice = self.get_bbox_image_patches(input_tensor, bbox_s, bbox_o)

        sub_queries = self.patch_embed(sub_masked_img)  # BNC
        obj_queries = self.patch_embed(obj_masked_img)
        sub_queries = sub_queries + self.pos_embed[:, self.num_tokens:, :] + self.sub_indicate_embed
        obj_queries = obj_queries + self.pos_embed[:, self.num_tokens:, :] + self.obj_indicate_embed

        sub_queries = self.blocks(sub_queries)
        obj_queries = self.blocks(obj_queries)
        sub_queries = self.norm(sub_queries)
        obj_queries = self.norm(obj_queries)

        input_query = torch.cat([sub_queries, obj_queries], dim=1)  # (B, 2N, embed_dim)

        # decode
        sub_obj_feature = self.decoder(input_query, image_ctx_tokens)  # (B, 2N, embed_dim)

        sub_feature = self.feature_pooling(sub_obj_feature[:, :self.num_patches, :], sub_img_mask_slice)  # (B, embed_dim)
        obj_feature = self.feature_pooling(sub_obj_feature[:, self.num_patches:, :], obj_img_mask_slice)

        feature = torch.cat([sub_feature, obj_feature], dim=1)
        return feature

    def forward_head(self, x, predicate):
        rel_dists = self.head(x)
        predi_onehot = F.one_hot(predicate, num_classes=self.predicate_dim).to(torch.float32)
        return torch.sum(rel_dists * predi_onehot, 1)

    def forward(self, full_im, bbox_s, bbox_o, predicate):
        x = self.forward_features(full_im, bbox_s, bbox_o, predicate)
        output = self.forward_head(x, predicate)
        return output
