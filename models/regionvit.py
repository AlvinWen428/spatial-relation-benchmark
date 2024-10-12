import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.ops.roi_align import RoIAlign

import models.modules.vision_transformer as vision_transformer


def union_bbox(bbox_a, bbox_b):
    return torch.cat([torch.min(bbox_a[:, 0], bbox_b[:, 0]).unsqueeze(1),
                      torch.max(bbox_a[:, 1], bbox_b[:, 1]).unsqueeze(1),
                      torch.min(bbox_a[:, 2], bbox_b[:, 2]).unsqueeze(1),
                      torch.max(bbox_a[:, 3], bbox_b[:, 3]).unsqueeze(1)], dim=1)


def bbox_unnormalize(bbox: torch.Tensor, image_h, image_w):
    # bbox: (B, 4)
    new_bbox = bbox.clone()
    new_bbox[:, 0] *= image_h
    new_bbox[:, 1] *= image_h
    new_bbox[:, 2] *= image_w
    new_bbox[:, 3] *= image_w
    return new_bbox


def bbox_resize(bbox, origin_h, origin_w, new_h, new_w):
    ratio_h = new_h / origin_h
    ratio_w = new_w / origin_w
    new_bbox = torch.cat(
        [
            (bbox[:, 0] * ratio_h).unsqueeze(1),
            (bbox[:, 1] * ratio_h).unsqueeze(1),
            (bbox[:, 2] * ratio_w).unsqueeze(1),
            (bbox[:, 3] * ratio_w).unsqueeze(1)
        ],
        dim=1
    )
    return new_bbox


def layer_init(layer, init_para=0.1, normal=False):
    xavier = False if normal == True else True
    if normal:
        torch.nn.init.normal_(layer.weight, mean=0, std=init_para)
        torch.nn.init.constant_(layer.bias, 0)
        return
    elif xavier:
        torch.nn.init.xavier_normal_(layer.weight, gain=1.0)
        torch.nn.init.constant_(layer.bias, 0)
        return


class RoiBoxFeatureExtractor(nn.Module):
    def __init__(self, in_channels, resolution, scale, sampling_ratio, representation_size):
        super(RoiBoxFeatureExtractor, self).__init__()
        self.pooler = RoIAlign(output_size=(resolution, resolution), spatial_scale=scale, sampling_ratio=sampling_ratio)
        input_size = in_channels * resolution ** 2
        self.fc = nn.Sequential(nn.Linear(input_size, representation_size),
                                nn.ReLU(),
                                nn.BatchNorm1d(representation_size),
                                )

    def forward_pool(self, x, bbox):
        # transfer the xxyy to xyxy first
        bbox_xyxy = bbox[:, [0, 2, 1, 3]]
        # concat the batch index on the first dimension
        bbox_xyxy = torch.cat((torch.arange(x.shape[0]).to(x.device).unsqueeze(1), bbox_xyxy), dim=1)
        x = self.pooler(x, bbox_xyxy)
        return x

    def forward_without_pool(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, x, bbox):
        x = self.forward_pool(x, bbox)
        x = self.forward_without_pool(x)
        return x


class UnionBoxFeatureExtractor(nn.Module):
    def __init__(self, in_channels, resolution, scale, sampling_ratio, representation_size):
        super(UnionBoxFeatureExtractor, self).__init__()
        self.feature_extractor = RoiBoxFeatureExtractor(in_channels, resolution, scale, sampling_ratio, representation_size)

        # union rectangle size
        self.rect_size = resolution * 4 - 1
        self.rect_conv = nn.Sequential(*[
            nn.Conv2d(2, in_channels // 2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels // 2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels, momentum=0.01),
        ])

    def get_rect_inputs(self, images, bbox_s, bbox_o):
        image_h, image_w = images.shape[2], images.shape[3]
        device = images.device
        num_rel = bbox_s.shape[0]
        # use range to construct rectangle, sized (rect_size, rect_size)
        dummy_x_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
        dummy_y_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
        # resize bbox to the scale rect_size
        head_bbox = bbox_resize(bbox_s, image_h, image_w, self.rect_size, self.rect_size)
        tail_bbox = bbox_resize(bbox_o, image_h, image_w, self.rect_size, self.rect_size)
        # here, the bbox is (x1, x2, y1, y2)
        head_rect = ((dummy_x_range >= head_bbox[:, 0].floor().view(-1, 1, 1).long()) & \
                     (dummy_x_range <= head_bbox[:, 1].ceil().view(-1, 1, 1).long()) & \
                     (dummy_y_range >= head_bbox[:, 2].floor().view(-1, 1, 1).long()) & \
                     (dummy_y_range <= head_bbox[:, 3].ceil().view(-1, 1, 1).long())).float()
        tail_rect = ((dummy_x_range >= tail_bbox[:, 0].floor().view(-1, 1, 1).long()) & \
                     (dummy_x_range <= tail_bbox[:, 1].ceil().view(-1, 1, 1).long()) & \
                     (dummy_y_range >= tail_bbox[:, 2].floor().view(-1, 1, 1).long()) & \
                     (dummy_y_range <= tail_bbox[:, 3].ceil().view(-1, 1, 1).long())).float()
        rect_input = torch.stack((head_rect, tail_rect), dim=1)
        return rect_input

    def forward(self, images, feature, bbox_s, bbox_o):
        union_bboxes = union_bbox(bbox_s, bbox_o)

        rect_inputs = self.get_rect_inputs(images, bbox_s, bbox_o)
        rect_features = self.rect_conv(rect_inputs)

        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_vis_features = self.feature_extractor.forward_pool(feature, union_bboxes)

        # merge two parts
        union_features = union_vis_features + rect_features
        union_features = self.feature_extractor.forward_without_pool(union_features)  # (total_num_rel, out_channels)
        return union_features


class RegionViT(nn.Module):
    def __init__(self, hidden_dim, predicate_dim, box_feature_size,
                 backbone="resnet18", pretrain_ckp=""):
        super(RegionViT, self).__init__()
        self.hidden_dim = hidden_dim
        self.predicate_dim = predicate_dim
        if backbone.startswith("resnet"):
            self.backbone = models.__dict__[backbone](pretrained=False)
            self.load_pretrained_resnet(backbone, pretrain_ckp)
            self.backbone = nn.Sequential(
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool,
                self.backbone.layer1,
                self.backbone.layer2,
                self.backbone.layer3,
                self.backbone.layer4,
            )
            backbone_feature_channel = {'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnet101': 2048}.get(backbone)
            pooler_scale = 1 / 32
        elif backbone.startswith("vit"):
            self.backbone = vision_transformer.__dict__[backbone](pretrain_ckp=pretrain_ckp)
            backbone_feature_channel = self.backbone.embed_dim
            pooler_scale = 1 / 16
        else:
            raise NotImplementedError

        self.box_feature_size = box_feature_size
        self.subject_feature_extractor = RoiBoxFeatureExtractor(backbone_feature_channel, resolution=3, scale=pooler_scale,
                                                                sampling_ratio=-1, representation_size=box_feature_size)
        self.object_feature_extractor = RoiBoxFeatureExtractor(backbone_feature_channel, resolution=3, scale=pooler_scale,
                                                               sampling_ratio=-1, representation_size=box_feature_size)
        self.union_feature_extractor = UnionBoxFeatureExtractor(backbone_feature_channel, resolution=3, scale=pooler_scale,
                                                                sampling_ratio=-1, representation_size=self.hidden_dim)

        self.pair_roi_embedding = nn.Sequential(*[nn.Linear(box_feature_size * 2, self.hidden_dim),
                                                  nn.ReLU(),
                                                  nn.BatchNorm1d(self.hidden_dim)])
        self.rel_classifier = nn.Sequential(*[nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                                              nn.ReLU(),
                                              nn.BatchNorm1d(self.hidden_dim),
                                              nn.Linear(self.hidden_dim, self.predicate_dim)])

    def load_pretrained_resnet(self, backbone_name, pretrained_ckp):
        if pretrained_ckp == "":
            pass
        elif pretrained_ckp == "supervised":
            self.backbone = models.__dict__[backbone_name](pretrained=True)
        else:
            checkpoint = torch.load(pretrained_ckp)
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            state_dict = {k.replace('module.', '').replace('encoder.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, full_im, bbox_s, bbox_o, predicate):
        image_feature = self.backbone(full_im)

        img_h, img_w = full_im.shape[2], full_im.shape[3]
        rescaled_bbox_s = bbox_unnormalize(bbox_s, img_h, img_w)
        rescaled_bbox_o = bbox_unnormalize(bbox_o, img_h, img_w)

        # roi features
        sub_feature = self.subject_feature_extractor(image_feature, rescaled_bbox_s)
        obj_feature = self.object_feature_extractor(image_feature, rescaled_bbox_o)

        # # union features
        union_feature = self.union_feature_extractor(full_im, image_feature, rescaled_bbox_s, rescaled_bbox_o)

        pair_roi_feature = self.pair_roi_embedding(torch.cat([sub_feature, obj_feature], dim=1))
        prod_rep = torch.cat((pair_roi_feature, union_feature), dim=1)
        rel_dists = self.rel_classifier(prod_rep)

        predi_onehot = F.one_hot(predicate, num_classes=self.predicate_dim).to(torch.float32)
        return torch.sum(rel_dists * predi_onehot, 1)
