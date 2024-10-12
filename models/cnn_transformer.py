from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import models.modules.vision_transformer as vision_transformer
from models.regionvit import RoiBoxFeatureExtractor, bbox_unnormalize
from models.modules.read_net import PromptReadoutNet


class CNNTransformer(nn.Module):
    def __init__(self, backbone="resnet50", pretrained="", pretrained_resnet="", predicate_dim=30,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.0,
                 use_attn_mask=False, roi_pool_size=3,
                 readnet_d_hidden=512, readnet_dropout=0.0,
                 prompt_emb_pool='max',
                 norm_layer=None, act_layer=None, init_std=0.02, *args, **kwargs):
        super(CNNTransformer, self).__init__()
        self.predicate_dim = predicate_dim
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.use_attn_mask = use_attn_mask
        self.roi_pool_size = roi_pool_size
        self.prompt_emb_pool = prompt_emb_pool
        self.init_std = init_std

        self.backbone_name = backbone
        if backbone.startswith("resnet"):
            self.backbone = self.build_backbone(backbone, pretrained_resnet)
            backbone_feature_channel = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnet101': 2048}.get(backbone)
            roi_align_input_channel = embed_dim
            pooler_scale = 1 / 32
            self.vis_feature_dim = (7, 7)
            self.num_patches = 7 * 7
        elif backbone.startswith("vit"):
            self.backbone = vision_transformer.__dict__[backbone](pretrain_ckp=pretrained)
            backbone_feature_channel = self.backbone.embed_dim
            roi_align_input_channel = self.backbone.embed_dim
            pooler_scale = 1 / 16
            self.vis_feature_dim = (14, 14)
            self.num_patches = 196
        else:
            raise NotImplementedError

        self.num_tokens = 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = nn.Conv2d(backbone_feature_channel, embed_dim, kernel_size=(1, 1))
        self.patch_embed_norm = norm_layer(embed_dim)

        self.subject_feature_extractor = RoiBoxFeatureExtractor(roi_align_input_channel, resolution=3,
                                                                scale=pooler_scale,
                                                                sampling_ratio=-1, representation_size=embed_dim)
        self.object_feature_extractor = RoiBoxFeatureExtractor(roi_align_input_channel, resolution=3,
                                                               scale=pooler_scale,
                                                               sampling_ratio=-1, representation_size=embed_dim)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + self.num_tokens, embed_dim)
        )
        # self.num_prompts = self.roi_pool_size ** 2
        self.num_prompts = 1

        # subject and object distinguishing embeddings
        self.subj_indicate_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.obj_indicate_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        vision_transformer.trunc_normal_(self.subj_indicate_embed, std=self.init_std)
        vision_transformer.trunc_normal_(self.obj_indicate_embed, std=self.init_std)

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                vision_transformer.Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    init_values=None,
                    beit_qkv_bias=False,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        readout_head_input_dim = self.embed_dim * 2
        self.readout_head = PromptReadoutNet(readout_head_input_dim, readnet_d_hidden, self.predicate_dim, readnet_dropout)

    def build_backbone(self, backbone_name, pretrained_resnet):
        backbone = models.__dict__[backbone_name](pretrained=False)

        if pretrained_resnet == '':
            pass
        elif pretrained_resnet == 'supervised':
            backbone = models.__dict__[backbone_name](pretrained=True)
        else:
            checkpoint = torch.load(pretrained_resnet)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', '').replace('encoder.', ''): v for k, v in state_dict.items()}
            load_info = backbone.load_state_dict(state_dict, strict=False)
            print(load_info)

        backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        return backbone

    def prompt_pooling(self, prompt_embedding):
        if self.prompt_emb_pool == 'max':
            pooled_embedding = torch.max(prompt_embedding, dim=1).values
        elif self.prompt_emb_pool == 'avg':
            pooled_embedding = torch.mean(prompt_embedding, dim=1)
        else:
            raise ValueError
        return pooled_embedding

    def forward_features(self, full_im, bbox_s, bbox_o, predicate):
        batch_size = full_im.shape[0]
        img_h, img_w = full_im.shape[2], full_im.shape[3]
        rescaled_bbox_s = bbox_unnormalize(bbox_s, img_h, img_w)
        rescaled_bbox_o = bbox_unnormalize(bbox_o, img_h, img_w)

        x = self.backbone(full_im)

        if self.backbone_name.startswith('resnet'):
            x = self.patch_embed(x)
            x = x.permute(0, 2, 3, 1)  # BHWC
            x = self.patch_embed_norm(x)
            x = x.permute(0, 3, 1, 2)  # BCHW

        sub_prompts = self.subject_feature_extractor(x, rescaled_bbox_s)
        obj_prompts = self.object_feature_extractor(x, rescaled_bbox_o)

        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        sub_prompts = sub_prompts.unsqueeze(1)
        obj_prompts = obj_prompts.unsqueeze(1)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed[:, :self.num_tokens, :]
        x = x + self.pos_embed[:, self.num_tokens:, :]

        sub_prompts = sub_prompts + self.pos_embed[:, self.num_tokens:self.num_tokens+self.num_prompts, :] + self.subj_indicate_embed
        obj_prompts = obj_prompts + self.pos_embed[:, self.num_tokens:self.num_tokens+self.num_prompts, :] + self.obj_indicate_embed
        x = torch.cat((cls_tokens, x, sub_prompts, obj_prompts), dim=1)

        # get attention mask
        n_patches = x.shape[1]
        attn_mask = torch.zeros(n_patches, n_patches).to(x.device)
        if self.use_attn_mask:
            attn_mask[:(self.num_tokens + self.num_patches), (self.num_tokens + self.num_patches):] = -float("inf")
            attn_mask.fill_diagonal_(0)

        for block_idx, block in enumerate(self.blocks):
            x = block(x, attn_mask=attn_mask)

        x = self.norm(x)

        sub_embedding = x[:, self.num_tokens + self.num_patches:self.num_tokens + self.num_patches + self.num_prompts,:]
        obj_embedding = x[:, self.num_tokens + self.num_patches + self.num_prompts:, :]

        sub_embedding = self.prompt_pooling(sub_embedding)
        obj_embedding = self.prompt_pooling(obj_embedding)

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

