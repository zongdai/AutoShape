# Copyright (c) 2021 Zongdai Liu. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import csv
import math
import logging
import random
import yaml
import paddle
import numpy as np
from PIL import Image
import pycocotools.coco as coco
from autoshape.utils.image import flip, color_aug
from autoshape.utils.image import get_affine_transform, affine_transform
from autoshape.utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from autoshape.utils.image import draw_dense_reg
from autoshape.cvlibs import manager
from autoshape.transforms import Compose



@manager.DATASETS.add_component
class Kitti_dataset(paddle.io.Dataset):
    """Parsing KITTI format dataset

    Args:
        Dataset (class):
    """
    def __init__(self, opt, split):
        super().__init__()
        self.num_class = opt.num_class
        self.num_keypoints = opt.num_keypoints
        self.max_objs = opt.max_objs

        self.data_dir = os.path.join(opt.data_dir, 'kitti')
        self.img_dir = os.path.join(self.data_dir, 'images')

        self.annot_path = os.path.join(self.data_dir, 'annotations', 'kitti_{}_{}.json').format(split, self.num_keypoints)

        self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)

        self.split = split
        self.opt = opt
        self.alpha_in_degree = False

        print('==> initializing kitti{} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        image_ids = self.coco.getImgIds()
        if 'train' in split:
            self.images = []
            for img_id in image_ids:
                idxs = self.coco.getAnnIds(imgIds=[img_id])
                if len(idxs) > 0:
                    self.images.append(img_id)
        else:
            self.images = image_ids
        self.num_samples = len(self.images)
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def __len__(self):
        return self.num_samples

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _convert_alpha(self, alpha):
        return math.radians(alpha + 45) if self.alpha_in_degree else alpha
    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        s = max(img.shape[0], img.shape[1]) * 1.0
        rot = 0
        flipped = False
        # if 'train' in self.split:
        #     if not self.opt.not_rand_crop:
        #         s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
        #         w_border = self._get_border(128, img.shape[1])
        #         h_border = self._get_border(128, img.shape[0])
        #         c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
        #         c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
        #     else:
        #         sf = self.opt.scale
        #         cf = self.opt.shift
        #         c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        #         c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
        #         s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

        trans_input = get_affine_transform(c, s, rot, [self.opt.input_w, self.opt.input_h])
        inp = cv2.warpAffine(img, trans_input, (self.opt.input_w, self.opt.input_h), flags=cv2.INTER_LINEAR)

        inp = (inp.astype(np.float32) / 255.)

        # if 'train' in self.split and not self.opt.no_color_aug:
        #     color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        # print(inp.shape)
        # cv2.imshow('t', inp)
        # cv2.imshow('ori', img)
        # cv2.waitKey()
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)
        num_keypoints = self.num_keypoints
        trans_output = get_affine_transform(c, s, 0, [self.opt.output_w, self.opt.output_h])
        trans_output_inv = get_affine_transform(c, s, 0, [self.opt.output_w, self.opt.output_h], inv=1)
        # object class and position in heatmap
        hm = np.zeros((self.num_class, self.opt.output_h, self.opt.output_w), dtype=np.float32)
        # ----- remove ------
        # hm_hp = np.zeros((num_keypoints, self.opt.output_h, self.opt.output_w), dtype=np.float32)
        # dense_kps = np.zeros((num_keypoints, 2, self.opt.output_h, self.opt.output_w),
        #                      dtype=np.float32)
        # dense_kps_mask = np.zeros((num_keypoints, self.opt.output_h, self.opt.output_w),
        #                           dtype=np.float32)
        # ----- remove ------
        opinv = np.zeros((self.max_objs, 2, 3), dtype=np.float32)
        opinv[:] = trans_output_inv
        # width and height for 2d bbox
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)

        # height, width, length for 3d bbox
        dim = np.zeros((self.max_objs, 3), dtype=np.float32)

        # x ,y, z coordinate
        location = np.zeros((self.max_objs, 3), dtype=np.float32)

        # z coordinate
        dep = np.zeros((self.max_objs, 1), dtype=np.float32)
        dep_mask = np.zeros((self.max_objs), dtype=np.float32)
        # yaw
        ori = np.zeros((self.max_objs, 1), dtype=np.float32)

        #
        rotbin = np.zeros((self.max_objs, 2), dtype=np.float32)

        #
        rotres = np.zeros((self.max_objs, 2), dtype=np.float32)

        # when rot_mask set dtype to np.int64, it gets error.
        rot_mask = np.zeros((self.max_objs, 1), dtype=np.float32)

        # 2d keypoints
        kps_2d = np.zeros((self.max_objs, num_keypoints * 2), dtype=np.float32)

        #
        kps_cent = np.zeros((self.max_objs, 2), dtype=np.float32)

        #
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)

        # index for locating object in heatmap for supervision
        ind = np.zeros((self.max_objs), dtype=np.int64)
        #
        reg_mask = np.zeros((self.max_objs), dtype=np.float32)
        wh_reg_mask = np.zeros((self.max_objs, 2), dtype=np.float32)
        dim_reg_mask = np.zeros((self.max_objs, 3), dtype=np.float32)
        p3d_reg_mask = np.zeros((self.max_objs, self.num_keypoints * 3), dtype=np.float32)
        #
        inv_mask = np.zeros((self.max_objs, self.num_keypoints * 2), dtype=np.float32)
        # mask for keypoints
        kps_mask = np.zeros((self.max_objs, self.num_keypoints * 2), dtype=np.float32)
        #
        coor_kps_mask = np.zeros((self.max_objs, self.num_keypoints * 2), dtype=np.float32)

        hp_offset = np.zeros((self.max_objs * num_keypoints, 2), dtype=np.float32)
        hp_ind = np.zeros((self.max_objs * num_keypoints), dtype=np.float32)
        hp_mask = np.zeros((self.max_objs * num_keypoints), dtype=np.float32)
        rot_scalar = np.zeros((self.max_objs, 1), dtype=np.float32)

        draw_gaussian = draw_umich_gaussian

        kps_3d = np.zeros((self.max_objs, self.num_keypoints * 3), dtype=np.float32)
        # p3ds_mask = np.zeros((self.max_objs, self.num_keypoints * 3), dtype=np.int64)

        calib = np.array(anns[0]['calib'], dtype=np.float32)
        calib = np.reshape(calib, (3, 4))
        gt_det = []
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(ann['category_id']) - 1
            pts = np.array(ann['keypoints'][:num_keypoints * 3], np.float32).reshape(num_keypoints, 3)
            alpha1 = ann['alpha']
            orien = ann['rotation_y']
            loc = ann['location']

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.opt.output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.opt.output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if (h > 0 and w > 0) or (rot != 0):
                alpha = self._convert_alpha(alpha1)
                if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
                    rotbin[k, 0] = 1
                    rotres[k, 0] = alpha - (-0.5 * np.pi)
                if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
                    rotbin[k, 1] = 1
                    rotres[k, 1] = alpha - (0.5 * np.pi)
                rot_scalar[k] = alpha
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * self.opt.output_w + ct_int[0]
                reg[k] = ct - ct_int
                dim[k] = ann['dim']
                kps_3d[k] = ann['p3d']

                # dim[k][0]=math.log(dim[k][0]/1.63)
                # dim[k][1] = math.log(dim[k][1]/1.53)
                # dim[k][2] = math.log(dim[k][2]/3.88)
                dep[k] = loc[2]
                dep_mask[k] = loc[2]
                ori[k] = orien
                location[k] = loc
                reg_mask[k] = 1
                wh_reg_mask[k, :] = 1
                dim_reg_mask[k, :] = 1
                p3d_reg_mask[k, :] = 1
                num_kpts = pts[:, 2].sum()
                if num_kpts == 0:
                    hm[cls_id, ct_int[1], ct_int[0]] = 0.9999
                    reg_mask[k] = 0
                    dim_reg_mask[k, :] = 0
                    p3d_reg_mask[k, :] = 0
                    wh_reg_mask[k, :] = 0
                rot_mask[k, :] = 1
                kps_cent[k, :] = pts[num_keypoints - 1, :2]
                for j in range(num_keypoints):
                    pts[j, :2] = affine_transform(pts[j, :2], trans_output)
                    kps_2d[k, j * 2: j * 2 + 2] = pts[j, :2] - ct_int
                    kps_mask[k, j * 2: j * 2 + 2] = 1

                if coor_kps_mask[k, (num_keypoints - 1) * 2] == 0 or coor_kps_mask[k, (num_keypoints - 1) * 2 + 1] == 0:
                    coor_kps_mask[k, :] = coor_kps_mask[k, :] * 0
                draw_gaussian(hm[cls_id], ct_int, radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1] +
                              pts[:, :2].reshape(num_keypoints * 2).tolist() + [cls_id])
        if rot != 0:
            hm = hm * 0 + 0.9999
            reg_mask *= 0
            dim_reg_mask *= 0
            p3d_reg_mask *= 0
            wh_reg_mask *= 0
            kps_mask *= 0
        meta = {'file_name': file_name}
        if flipped:
            coor_kps_mask = coor_kps_mask * 0
            inv_mask = inv_mask * 0

        f = np.ones((self.max_objs, num_keypoints * 2), dtype=np.float32) * calib[0][0]
        cxy = np.ones((self.max_objs, num_keypoints * 2), dtype=np.float32)
        cxy[..., ::2] = calib[0][2]
        cxy[..., 1::2] = calib[1][2]
        dep_mask[dep_mask < 5] = dep_mask[dep_mask < 5] * 0.01
        dep_mask[dep_mask >= 5] = np.log10(dep_mask[dep_mask >= 5] - 4) + 0.1
        ret = {
               'input': inp,
               'hm': hm,
               'reg_mask': reg_mask,
               'wh_reg_mask': wh_reg_mask,
               'dim_reg_mask': dim_reg_mask,
               'p3d_reg_mask': p3d_reg_mask,
               'ind': ind,
               'wh': wh,
               'hps': kps_2d,
               'hps_mask': kps_mask,
               'dim': dim,
               'rotbin': rotbin,
               'rotres': rotres,
               'rot_mask': rot_mask,
               'dep': dep,
               'dep_mask': dep_mask,
               'rotscalar': rot_scalar,
               'kps_cent': kps_cent,
               'calib': calib,
               'trans_output_inv': trans_output_inv,
               'opinv': opinv,
               # 'meta': meta,
               # "label_sel":label_sel,
               'location': location,
               'ori': ori,
               'coor_kps_mask': coor_kps_mask,
               'inv_mask': inv_mask,
               # 'p3ds_mask': p3ds_mask,
               'p3d': kps_3d,
               'f': f,
               'cxy': cxy
               }
        # if self.opt.debug > 0 or not 'train' in self.split:
        #     gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
        #         np.zeros((1, 40), dtype=np.float32)
        #     meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
        #     ret['meta'] = meta
        # for k, v in ret.items():
        #     print(v.shape)
        return ret

