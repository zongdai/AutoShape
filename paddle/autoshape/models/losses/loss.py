# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import numpy as np
import cv2
import paddle
import paddle.nn as nn
from paddle.nn import functional as F
from autoshape.ops.gather import transpose_and_gather_feat




class RegP2DL1Loss(nn.Layer):
    '''
    References from KM3D
    '''
    def __init__(self):
        super(RegP2DL1Loss, self).__init__()

    def forward(self, output, mask, ind, target, dep):
        pred = transpose_and_gather_feat(output, ind)
        loss = paddle.abs(pred * mask-target * mask)
        loss = paddle.sum(loss, axis=2) * dep
        loss = paddle.sum(loss)
        loss = loss / (paddle.sum(mask) + 1e-4)
        return loss

class RegP3DL1Loss(nn.Layer):
    def __init__(self):
        super(RegP3DL1Loss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = transpose_and_gather_feat(output, ind)
        loss = F.l1_loss(pred * mask, target * mask,  reduction='sum')
        loss = loss / (paddle.sum(mask) + 1e-4)
        return loss

class BinRotLoss(nn.Layer):
    def __init__(self):
        super(BinRotLoss, self).__init__()

    def forward(self, output, mask, ind, rotbin, rotres):
        pred = transpose_and_gather_feat(output, ind)
        loss = compute_rot_loss(pred, rotbin, rotres, mask)
        return loss

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos,
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = paddle.reshape(output, (-1, 8))
    target_bin = target_bin.astype(paddle.int64)
    target_bin = paddle.reshape(target_bin, (-1, 2))
    target_res = paddle.reshape(target_res, (-1, 2))
    mask = paddle.reshape(mask, (-1, 1))
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)

    loss_res = paddle.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        # idx1 = target_bin[:, 0].nonzero()[:, 0]
        idx1 = paddle.nonzero(target_bin[:, 0])[:, 0]
        valid_output1 = paddle.index_select(output, idx1, 0)
        valid_target_res1 = paddle.index_select(target_res, idx1, 0)
        loss_sin1 = compute_res_loss(
            valid_output1[:, 2], paddle.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
            valid_output1[:, 3], paddle.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        # idx2 = target_bin[:, 1].nonzero()[:, 0]
        idx2 = paddle.nonzero(target_bin[:, 1])[:, 0]

        valid_output2 = paddle.index_select(output, idx2, 0)
        valid_target_res2 = paddle.index_select(target_res, idx2, 0)
        loss_sin2 = compute_res_loss(
            valid_output2[:, 6], paddle.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
            valid_output2[:, 7], paddle.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res
    # return loss_bin1 + loss_bin2

def compute_bin_loss(output, target, mask):
    mask = paddle.expand_as(mask, output)
    output = output * mask
    return F.cross_entropy(output, target, reduction='mean')

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='mean')

class PositionLoss(nn.Layer):

    def __init__(self, opt):
        super(PositionLoss, self).__init__()

        self.num_joints = opt.num_keypoints
        self.opt = opt
        self.l1 = paddle.nn.loss.L1Loss(reduction='sum')

    def forward(self, output, batch):

        dim = transpose_and_gather_feat(output['dim'], batch['ind'])
        rot = transpose_and_gather_feat(output['rot'], batch['ind'])
        # prob = transpose_and_gather_feat(output['prob'], batch['ind'])
        kps = transpose_and_gather_feat(output['hps'], batch['ind'])
        p3d = transpose_and_gather_feat(output['p3d'], batch['ind'])


        kps_mask = batch['hps_mask'].astype(paddle.float32)
        calib = batch['calib']
        opinv = batch['opinv']  # batch_size x num_obj x 2 x 3

        num_object = dim.shape[1]
        batch_size = dim.shape[0]
        num_joints = self.num_joints

        # Camera
        f = batch['f']
        cxy = batch['cxy']

        cys = (batch['ind'] / self.opt.output_w).astype(paddle.float32)
        cxs = (batch['ind'] % self.opt.output_w).astype(paddle.float32)
        cxs = paddle.expand(cxs, shape=[num_joints, batch_size, num_object])
        cxs = paddle.transpose(cxs, perm=[1, 2, 0])
        cys = paddle.expand(cys, shape=[num_joints, batch_size, num_object])
        cys = paddle.transpose(cys, perm=[1, 2, 0])
        kps[:, :, ::2] = kps[:, :, ::2] + cxs
        kps[:, :, 1::2] = kps[:, :, 1::2] + cys

        kps = paddle.reshape(kps, (batch_size, num_object, num_joints, 2))
        kps = paddle.transpose(kps, perm=[0, 1, 3, 2])
        hom = paddle.ones((batch_size, num_object, 1, num_joints), dtype=paddle.float32)
        kps = paddle.concat([kps, hom], axis=2)
        kps = paddle.reshape(kps, (batch_size*num_object, 3, num_joints))
        opinv = paddle.reshape(opinv, (batch_size*num_object, 2, 3))
        kps = paddle.bmm(opinv, kps)
        kps = paddle.reshape(kps, (batch_size, num_object, 2, num_joints))
        kps = paddle.transpose(kps, perm=[0, 1, 3, 2])
        kps = paddle.reshape(kps, (batch_size, num_object, 2*num_joints))


        kps_norm = (kps - cxy) / f  # batch_size x num_obj x (2*num_joints)
        kps_norm = paddle.unsqueeze(kps_norm, 3)  # batch_size x num_obj x (2*num_joints) x 1

        # Rot decode
        si = paddle.zeros_like(kps[:, :, 0:1]) + f[:, :, 0:1]
        alpha_idx = rot[:, :, 1] > rot[:, :, 5]
        alpha_idx = alpha_idx.astype(paddle.float32)
        alpha1 = paddle.atan(rot[:, :, 2] / rot[:, :, 3]) + (-0.5 * np.pi)
        alpha2 = paddle.atan(rot[:, :, 6] / rot[:, :, 7]) + (0.5 * np.pi)
        alpna_pre = alpha1 * alpha_idx + alpha2 * (1 - alpha_idx)
        alpna_pre = paddle.unsqueeze(alpna_pre, 2)

        rot_y = alpna_pre + paddle.atan(
            (kps[:, :, (num_joints - 1) * 2:(num_joints - 1) * 2 + 1] - calib[:, 0:1, 2:3]) / si)
        # rot_y[rot_y > np.pi] = rot_y[rot_y > np.pi] - 2 * np.pi
        # rot_y[rot_y < - np.pi] = rot_y[rot_y < - np.pi] + 2 * np.pi

        # Build Linear System
        # given const
        const = paddle.to_tensor([[-1, 0, 0, -1]], stop_gradient=False)
        const = paddle.expand(const, shape=[num_joints, 4])
        const = paddle.reshape(const, (num_joints * 2, 2))
        # given const
        const = paddle.expand(const, shape=[batch_size, num_object, num_joints * 2, 2]).astype(
            paddle.float32)  # batch_size x num_obj x (2*num_joints) x 2
        A = paddle.concat([const, kps_norm], axis=3)  # batch_size x num_obj x (2*num_joints) x 3
        B = paddle.zeros_like(kps_norm).astype(paddle.float32)  # batch_size x num_obj x (2*num_joints) x 1
        C = paddle.zeros_like(kps_norm).astype(paddle.float32)  # batch_size x num_obj x (2*num_joints) x 1

        # Fill B and C
        dim = paddle.unsqueeze(dim, axis=3)
        p3d = paddle.unsqueeze(p3d, axis=3)
        rot_y = paddle.unsqueeze(rot_y, axis=3)
        cosori = paddle.cos(rot_y)
        sinori = paddle.sin(rot_y)

        l = dim[:, :, 2:3, :]
        h = dim[:, :, 0:1, :]
        w = dim[:, :, 1:2, :]
        for i in range((num_joints - 9)):
            B[:, :, i * 2:i * 2 + 1] = p3d[:, :, i * 3:i * 3 + 1] * cosori + p3d[:, :, i * 3 + 2:i * 3 + 3] * sinori
            B[:, :, i * 2 + 1:i * 2 + 2] = p3d[:, :, i * 3 + 1:i * 3 + 2]

        index_base = (num_joints - 9) * 2
        B[:, :, index_base + 0:index_base + 1] = l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, index_base + 1:index_base + 2] = h * 0.5
        B[:, :, index_base + 2:index_base + 3] = l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, index_base + 3:index_base + 4] = h * 0.5
        B[:, :, index_base + 4:index_base + 5] = -l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, index_base + 5:index_base + 6] = h * 0.5
        B[:, :, index_base + 6:index_base + 7] = -l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, index_base + 7:index_base + 8] = h * 0.5
        B[:, :, index_base + 8:index_base + 9] = l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, index_base + 9:index_base + 10] = -h * 0.5
        B[:, :, index_base + 10:index_base + 11] = l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, index_base + 11:index_base + 12] = -h * 0.5
        B[:, :, index_base + 12:index_base + 13] = -l * 0.5 * cosori - w * 0.5 * sinori
        B[:, :, index_base + 13:index_base + 14] = -h * 0.5
        B[:, :, index_base + 14:index_base + 15] = -l * 0.5 * cosori + w * 0.5 * sinori
        B[:, :, index_base + 15:index_base + 16] = -h * 0.5
        B[:, :, index_base + 16:index_base + 17] = 0
        B[:, :, index_base + 17:index_base + 18] = 0

        for i in range((num_joints - 9)):
            C[:, :, i * 2:i * 2 + 1] = -p3d[:, :, i * 3 + 0:i * 3 + 1] * sinori + p3d[:, :,
                                                                                  i * 3 + 2:i * 3 + 3] * cosori
            C[:, :, i * 2 + 1:i * 2 + 2] = -p3d[:, :, i * 3 + 0:i * 3 + 1] * sinori + p3d[:, :,
                                                                                      i * 3 + 2:i * 3 + 3] * cosori
        C[:, :, index_base + 0:index_base + 1] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, index_base + 1:index_base + 2] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, index_base + 2:index_base + 3] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, index_base + 3:index_base + 4] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, index_base + 4:index_base + 5] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, index_base + 5:index_base + 6] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, index_base + 6:index_base + 7] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, index_base + 7:index_base + 8] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, index_base + 8:index_base + 9] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, index_base + 9:index_base + 10] = -l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, index_base + 10:index_base + 11] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, index_base + 11:index_base + 12] = -l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, index_base + 12:index_base + 13] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, index_base + 13:index_base + 14] = l * 0.5 * sinori - w * 0.5 * cosori
        C[:, :, index_base + 14:index_base + 15] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, index_base + 15:index_base + 16] = l * 0.5 * sinori + w * 0.5 * cosori
        C[:, :, index_base + 16:index_base + 17] = 0
        C[:, :, index_base + 17:index_base + 18] = 0
        B = B - kps_norm * C

        A = paddle.reshape(A, (batch_size * num_object, num_joints * 2, 3))
        B = paddle.reshape(B, (batch_size * num_object, num_joints * 2, 1))
        AT = paddle.transpose(A, perm=[0, 2, 1])
        # ======Add Adaptive Weights=======
        # adaptive_weights = paddle.ones((batch_size, num_object, num_joints*2), dtype=paddle.float32)
        # adaptive_weights = paddle.reshape(adaptive_weights, (batch_size*num_object, num_joints*2))
        # sigmoid = paddle.nn.Sigmoid()
        # adaptive_weights = sigmoid(adaptive_weights)
        # print(adaptive_weights)
        # adaptive_weights_diag = paddle.diag(adaptive_weights)  # batch_size x num_obj x (num_joints * 2) x (num_joints * 2)
        # print(adaptive_weights_diag)
        # adaptive_weights_diag = adaptive_weights_diag.view(b * c, self.num_joints * 2, self.num_joints * 2)
        # A = torch.bmm(adaptive_weights_diag, A)
        # B = torch.bmm(adaptive_weights_diag, B)
        # AT = A.permute(0, 2, 1)
        #       ======END=======

        # Calculate
        loss_mask = paddle.sum(kps_mask, axis=2)  # kps_mask.shape = batch_size x num_obj x 2*num_joints
        loss_mask = loss_mask > 15
        pinv = paddle.bmm(AT, A)
        is_nan = paddle.isnan(pinv).astype(paddle.float32)
        if paddle.sum(is_nan) > 0:
            print('nan')
            return 0
        pinv = paddle.inverse(pinv)
        pinv = paddle.bmm(pinv, AT)
        pinv = paddle.bmm(pinv, B)
        pinv = paddle.reshape(pinv, (batch_size, num_object, 3))
        dim = paddle.squeeze(dim, 3)
        pinv[:, :, 1] = pinv[:, :, 1] + dim[:, :, 0] / 2

        # loss_location = F.l1_loss(pinv[:, :, 0]*loss_mask, batch['location'][:, :, 0]*loss_mask, reduction='sum') + \
        #                 F.l1_loss(pinv[:, :, 1]*loss_mask, batch['location'][:, :, 1]*loss_mask, reduction='sum') + \
        #                 F.l1_loss(pinv[:, :, 2]*loss_mask, batch['location'][:, :, 2]*loss_mask, reduction='sum')
        # # print(paddle.sum(loss_mask.astype(paddle.float32)))
        # loss_location = loss_location / (paddle.sum(loss_mask.astype(paddle.float32)) + 1e-4)

        loss = (pinv - batch['location'])
        loss_norm = paddle.norm(loss, p=2, axis=2)
        loss_location = loss_norm * loss_mask

        mask_num = paddle.sum(loss != 0).astype(paddle.float32)
        loss_location = paddle.sum(loss_location) / (mask_num + 1)

        return loss_location