import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import json
import random
import numpy as np

import os.path
import cv2
import math
from PIL import Image, ImageDraw


IMAGE_SIZE = 224


# TODO: implement the shift augmentation for SpatialSense dataset
class SpatialSenseDataset(Dataset):
    def __init__(self, split, predicate_dim, object_dim, data_path, load_img,
                 data_aug_shift, data_aug_color, crop, norm_data):
        super().__init__()
        self.split = split
        self.load_image = load_img
        self.annotation_path = data_path
        self.img_path = os.path.join(os.path.dirname(data_path), 'images/')
        with open(self.annotation_path, 'r') as f:
            self.data = json.load(f)

        self.predicates = self.data['predicates']
        self.objects = self.data['objects']
        assert len(self.predicates) == predicate_dim  # 9
        assert len(self.objects) == object_dim  # 3679

        self.data_aug_shift = data_aug_shift
        self.data_aug_color = data_aug_color
        self.crop = crop
        self.norm_data = norm_data

        self.im_mean = [0.485, 0.456, 0.406]
        self.im_std = [0.229, 0.224, 0.225]
        self.depth_mean = [0.3874085163399082]  # depth only have one channel
        self.depth_std = [0.26913277225836924]

        self.rgb_path_to_id = {}
        idx = 0
        self.annotations = []
        self.annot_idx_each_predicate = {k:[] for k in self.predicates}
        for img in self.data['sample_annots']:
            if img["split"] in split.split("_"):
                for annot in img["annotations"]:
                    annot["url"] = img["url"]
                    annot["height"] = img["height"]
                    annot["width"] = img["width"]
                    annot["subject"]["bbox"] = self.fix_bbox(
                        annot["subject"]["bbox"], img["height"], img["width"]
                    )
                    annot["object"]["bbox"] = self.fix_bbox(
                        annot["object"]["bbox"], img["height"], img["width"]
                    )
                    self.annotations.append(annot)
                    self.rgb_path_to_id[self.get_img_path(annot["url"], self.img_path)] = idx
                    self.annot_idx_each_predicate[annot['predicate']].append(idx)
                    idx += 1

        print("%d relations in %s" % (len(self.annotations), split))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx, visualize=False):
        annot = self.annotations[idx]

        t_s = self._getT(annot["subject"]["bbox"], annot["object"]["bbox"])
        t_o = self._getT(annot["object"]["bbox"], annot["subject"]["bbox"])

        ys0, ys1, xs0, xs1 = annot["subject"]["bbox"]
        yo0, yo1, xo0, xo1 = annot["object"]["bbox"]
        union_bbox = self._getUnionBBox(annot["subject"]["bbox"], annot["object"]["bbox"],
                                        annot["height"], annot["width"])

        datum = {
            "url": annot["url"],
            "_id": annot["_id"],
            "subject": {
                "name": annot["subject"]["name"],
                "idx": self.objects.index(annot['subject']['name']),
                "bbox": np.asarray(
                    [
                        ys0 / annot["height"],
                        ys1 / annot["height"],
                        xs0 / annot["width"],
                        xs1 / annot["width"],
                    ],
                    dtype=np.float32,
                ),
                "t": np.asarray(t_s, dtype=np.float32),
            },
            "object": {
                "name": annot["object"]["name"],
                "idx": self.objects.index(annot['subject']['name']),
                "bbox": np.asarray(
                    [
                        yo0 / annot["height"],
                        yo1 / annot["height"],
                        xo0 / annot["width"],
                        xo1 / annot["width"],
                    ],
                    dtype=np.float32,
                ),
                "t": np.asarray(t_o, dtype=np.float32),
            },
            "label": annot["label"],  # np.asarray([[annot["label"]]], dtype=np.float32),
            "predicate": {
                'name': annot["predicate"],
                'idx': self.predicates.index(annot["predicate"]),
                'bbox': np.asarray(
                    [
                        union_bbox[0] / annot["height"],
                        union_bbox[1] / annot["height"],
                        union_bbox[2] / annot["width"],
                        union_bbox[3] / annot["width"],
                    ], dtype=np.float32,
                )
            },
            'rgb_source': self.get_img_path(annot["url"], self.img_path),
        }

        if self.load_image:
            img = self.read_img(annot["url"], self.img_path)
            ih, iw = img.shape[:2]

            depth = self.read_depth(annot["url"], self.img_path)

            bbox_mask = np.stack(
                [
                    self._getDualMask(ih, iw, annot["subject"]["bbox"], ih, iw).astype(np.uint8),
                    self._getDualMask(ih, iw, annot["object"]["bbox"], ih, iw).astype(np.uint8),
                    np.zeros((ih, iw), dtype=np.uint8),
                ],
                axis=2,
            )

            if self.crop:
                enlarged_union_bbox = self.enlarge(union_bbox, 1.25, ih, iw, )
                full_img = Image.fromarray(img[enlarged_union_bbox[0]:enlarged_union_bbox[1], enlarged_union_bbox[2]:enlarged_union_bbox[3], :].astype(np.uint8, copy=False), mode="RGB")
                full_depth = Image.fromarray(depth[enlarged_union_bbox[0]:enlarged_union_bbox[1], enlarged_union_bbox[2]:enlarged_union_bbox[3]].astype(np.uint8, copy=False))  # gray image
                bbox_mask = Image.fromarray(bbox_mask[enlarged_union_bbox[0]:enlarged_union_bbox[1], enlarged_union_bbox[2]:enlarged_union_bbox[3], :].astype(np.uint8, copy=False), mode="RGB")
            else:
                full_img = Image.fromarray(img.astype(np.uint8, copy=False), mode="RGB")
                full_depth = Image.fromarray(depth.astype(np.uint8, copy=False))
                bbox_mask = Image.fromarray(bbox_mask.astype(np.uint8, copy=False), mode="RGB")

            if "train" in self.split:
                if self.data_aug_shift:
                    crop_top, crop_left, crop_h, crop_w = \
                        transforms.RandomResizedCrop.get_params(full_img, scale=[0.75, 0.85], ratio=[3. / 4., 4. / 3.])
                    full_img = TF.resized_crop(full_img, crop_top, crop_left, crop_h, crop_w, [IMAGE_SIZE, IMAGE_SIZE])
                    full_depth = TF.resized_crop(full_depth, crop_top, crop_left, crop_h, crop_w, [IMAGE_SIZE, IMAGE_SIZE])
                    bbox_mask = TF.resized_crop(bbox_mask, crop_top, crop_left, crop_h, crop_w, [IMAGE_SIZE, IMAGE_SIZE])
                else:
                    full_img = TF.resize(full_img, size=[IMAGE_SIZE, IMAGE_SIZE])
                    full_depth = TF.resize(full_depth, size=[IMAGE_SIZE, IMAGE_SIZE])
                    bbox_mask = TF.resize(bbox_mask, size=[IMAGE_SIZE, IMAGE_SIZE])
                if self.data_aug_color:
                    full_img = TF.adjust_brightness(full_img, random.uniform(0.9, 1.1))
                    full_img = TF.adjust_contrast(full_img, random.uniform(0.9, 1.1))
                    full_img = TF.adjust_gamma(full_img, random.uniform(0.9, 1.1))
                    full_img = TF.adjust_hue(full_img, random.uniform(-0.05, 0.05))
            else:
                full_img = TF.resize(full_img, size=[IMAGE_SIZE, IMAGE_SIZE])
                full_depth = TF.resize(full_depth, size=[IMAGE_SIZE, IMAGE_SIZE])
                bbox_mask = TF.resize(bbox_mask, size=[IMAGE_SIZE, IMAGE_SIZE])

            full_img = TF.to_tensor(full_img)
            full_depth = TF.to_tensor(full_depth)
            resized_bbox_mask = TF.resize(bbox_mask, [32, 32])
            resized_bbox_mask = TF.to_tensor(resized_bbox_mask)[:2].float() / 255.0
            bbox_mask = TF.to_tensor(bbox_mask)[:2].float() / 255.0

            if self.norm_data:
                full_img = TF.normalize(full_img, mean=self.im_mean, std=self.im_std)
                full_depth = TF.normalize(full_depth, mean=self.depth_mean, std=self.depth_std)

            datum["img"] = full_img
            datum["depth"] = full_depth

            datum['subject']['bbox'] = self.get_bbox_coord_from_mask(bbox_mask[0])
            datum['object']['bbox'] = self.get_bbox_coord_from_mask(bbox_mask[1])
            datum['predicate']['bbox'] = np.array(
                self._getUnionBBox(datum['subject']['bbox'], datum['object']['bbox'], ih=1, iw=1)
            )

            datum['bbox_mask'] = resized_bbox_mask

            if visualize:
                vis = Image.new(mode='RGB', size=(224 * 4, 224))
                raw_img = Image.fromarray(img.astype(np.uint8))
                raw_img = raw_img.resize(size=(224, 224))
                raw_depth = Image.fromarray(depth.astype(np.uint8))
                raw_depth = raw_depth.resize(size=(224, 224)).convert('RGB')

                input_img = ((full_img * torch.tensor(self.im_std).view(-1, 1, 1) + torch.tensor(self.im_mean).view(-1, 1, 1)) * 255).permute((1, 2, 0)).numpy().astype(np.uint8)
                input_img_obj = Image.fromarray(input_img)
                draw = ImageDraw.Draw(input_img_obj)
                draw.rectangle(((xs0 * 224 / iw, ys0 * 224 / ih), (xs1 * 224 / iw, ys1 * 224 / ih)), outline='blue')
                draw.rectangle(((xo0 * 224 / iw, yo0 * 224 / ih), (xo1 * 224 / iw, yo1 * 224 / ih)), outline='red')

                input_depth = Image.fromarray(((full_depth * torch.tensor(self.depth_std).view(-1, 1, 1) + torch.tensor(self.depth_mean).view(-1, 1, 1)) * 255).squeeze(0).numpy().astype(np.uint8))
                input_depth = input_depth.convert('RGB')

                vis.paste(raw_img, (0, 0))
                vis.paste(input_img_obj, (224, 0))
                vis.paste(raw_depth, (224 * 2, 0))
                vis.paste(input_depth, (224 * 3, 0))
                draw = ImageDraw.Draw(vis)
                draw.text(
                    (224, 0),
                    f"{annot['subject']['name']}"
                    f"--{annot['predicate']}"
                    f"--{annot['object']['name']}"
                    f"--{annot['label']}",
                    (255, 255, 255),
                )
                vis.save('vis_%04d.jpg' % idx)
                exit()

        return datum

    @staticmethod
    def get_img_path(url, imagepath):
        if url.startswith("http"):  # flickr
            filename = os.path.join(imagepath, "flickr", url.split("/")[-1])
        else:  # nyu
            filename = os.path.join(imagepath, "nyu", url.split("/")[-1])
        return filename

    def read_img(self, url, imagepath):
        filename = self.get_img_path(url, imagepath)
        img = cv2.imread(filename).astype(np.float32, copy=False)[:, :, ::-1]
        assert img.shape[2] == 3
        return img

    @staticmethod
    def get_depth_path(url, imagepath):
        if url.startswith("http"):  # flickr
            filename = os.path.join(imagepath, "flickr-depth", url.split("/")[-1])
        else:  # nyu
            filename = os.path.join(imagepath, "nyu-depth", url.split("/")[-1])
        return filename

    def read_depth(self, url, imagepath):
        filename = self.get_depth_path(url, imagepath)
        filename = filename[:-4] + '.png'
        depth_3channel = cv2.imread(filename).astype(np.float32, copy=False)[:, :, ::-1]
        assert depth_3channel.shape[2] == 3
        depth = depth_3channel.mean(axis=2)
        return depth

    def enlarge(self, bbox, factor, ih, iw):
        height = bbox[1] - bbox[0]
        width = bbox[3] - bbox[2]
        assert height > 0 and width > 0
        return [
            max(0, int(bbox[0] - (factor - 1.0) * height / 2.0)),
            min(ih, int(bbox[1] + (factor - 1.0) * height / 2.0)),
            max(0, int(bbox[2] - (factor - 1.0) * width / 2.0)),
            min(iw, int(bbox[3] + (factor - 1.0) * width / 2.0)),
        ]

    @staticmethod
    def _getAppr(im, bb, out_size=224.0):
        subim = im[bb[0] : bb[1], bb[2] : bb[3], :]
        subim = cv2.resize(
            subim,
            None,
            None,
            out_size / subim.shape[1],
            out_size / subim.shape[0],
            interpolation=cv2.INTER_LINEAR,
        )
        return subim.astype(np.uint8, copy=False)

    @staticmethod
    def _getUnionBBox(aBB, bBB, ih, iw, margin=10):
        return [
            max(0, min(aBB[0], bBB[0]) - margin),
            min(ih, max(aBB[1], bBB[1]) + margin),
            max(0, min(aBB[2], bBB[2]) - margin),
            min(iw, max(aBB[3], bBB[3]) + margin),
        ]

    @staticmethod
    def _getDualMask(ih, iw, bb, heatmap_h=32, heatmap_w=32):
        rh = float(heatmap_h) / ih
        rw = float(heatmap_w) / iw
        x1 = max(0, int(math.floor(bb[0] * rh)))
        x2 = min(heatmap_h, int(math.ceil(bb[1] * rh)))
        y1 = max(0, int(math.floor(bb[2] * rw)))
        y2 = min(heatmap_w, int(math.ceil(bb[3] * rw)))
        mask = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
        mask[x1:x2, y1:y2] = 255
        # assert(mask.sum() == (y2 - y1) * (x2 - x1))
        return mask

    @staticmethod
    def get_bbox_coord_from_mask(mask):
        assert mask.dim() == 2
        h, w = mask.shape
        nz_mask = mask.nonzero()
        if len(nz_mask) != 0:
            t, l = nz_mask.min(0).values
            b, r = nz_mask.max(0).values
        # this can happen when we crop an object out of the image
        else:
            # Resolution: there are some samples with height=0, width=0
            # this happens when object is right at the edge
            # assert self.data_aug_shift
            # assert self.split == "train"
            t, l, b, r = [0] * 4

        return np.array([
            float(t) / h, float(b) / h,
            float(l) / w, float(r) / w
        ], np.float32)

    @staticmethod
    def _getT(bbox1, bbox2):
        h1 = bbox1[1] - bbox1[0]
        w1 = bbox1[3] - bbox1[2]
        h2 = bbox2[1] - bbox2[0]
        w2 = bbox2[3] - bbox2[2]
        return [
            (bbox1[0] - bbox2[0]) / float(h2),
            (bbox1[2] - bbox2[2]) / float(w2),
            math.log(h1 / float(h2)),
            math.log(w1 / float(w2)),
        ]

    @staticmethod
    def fix_bbox(bbox, ih, iw):
        if bbox[1] - bbox[0] < 20:
            if bbox[0] > 10:
                bbox[0] -= 10
            if bbox[1] < ih - 10:
                bbox[1] += 10

        if bbox[3] - bbox[2] < 20:
            if bbox[2] > 10:
                bbox[2] -= 10
            if bbox[3] < iw - 10:
                bbox[3] += 1
        return bbox
