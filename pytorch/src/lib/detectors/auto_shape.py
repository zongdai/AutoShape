from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch

try:
    from external.nms import soft_nms_39
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import multi_pose_decode
from models.decode import autoshape_pose_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import multi_pose_post_process
from utils.post_process import autoshape_post_process
from utils.autoshape_debugger import AutoShapeDebugger

from .base_detector import BaseDetector


class AutoShapeDetector(BaseDetector):
    def __init__(self, opt):
        super(AutoShapeDetector, self).__init__(opt)
        self.flip_idx = opt.flip_idx
        self.num_joints = opt.num_joints
        const_item = torch.Tensor([[-1, 0],[0, -1]])
        self.const = const_item.repeat((self.num_joints, 1))
        self.is_use_corner = opt.is_use_corner
        self.debugger = AutoShapeDebugger(dataset=self.opt.dataset, ipynb=(self.opt.debug==3), theme=self.opt.debugger_theme)

    def process(self, images,meta, return_time=False):
        with torch.no_grad():
            torch.cuda.synchronize()
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            # if self.opt.hm_hp and not self.opt.mse_loss:
            #     output['hm_hp'] = output['hm_hp'].sigmoid_()
            #
            # reg = output['reg'] if self.opt.reg_offset else None
            # # print(self.opt.hm_hp)
            # hm_hp = output['hm_hp'] if self.opt.hm_hp else None
            # hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
            torch.cuda.synchronize()
            forward_time = time.time()


            dets = autoshape_pose_decode(output['hm'], output['hps'], output['dim'], output['rot'], output['p3d'], output['adaptive_weights'], prob=output['prob'],K=self.opt.K, meta=meta, const=self.const, is_use_corner=self.is_use_corner)


        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        dets = autoshape_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.num_joints)
        for j in range(1,2):#, self.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 1 + 4 + self.num_joints * 8 + 3 + 1 + 2 + 2 + 1)
            # import pdb; pdb.set_trace()
            dets[0][j][:, :4] /= scale
            dets[0][j][:, 5:5 + self.num_joints * 2] /= scale
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        results[1] = np.concatenate(
            [detection[1] for detection in detections], axis=0).astype(np.float32)
        if self.opt.nms or len(self.opt.test_scales) > 1:
            soft_nms_39(results[1], Nt=0.5, method=2)
        results[1] = results[1].tolist()
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:39] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
                               img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

    def show_results(self, debugger, image, results, calib):
        debugger.add_img(image, img_id='car_pose')
        for i, bbox in enumerate(results[1]):
            # print(len(bbox))
            if bbox[4] > self.opt.vis_thresh:
                # debugger.add_coco_bbox(bbox[:4], bbox[40], bbox[4], img_id='car_pose')
                # debugger.add_kitti_hp(bbox[5:5 + 25 * 2], img_id='car_pose')
                # debugger.add_kitti_hp(bbox[5:55], img_id='car_pose')
                # debugger.add_bev(bbox, img_id='car_pose',is_faster=self.opt.faster)
                debugger.save_kitti_format(bbox, self.image_path, self.opt, is_faster=self.opt.faster)
                # debugger.add_3d_detection(bbox, self.image_path, self.opt, i, calib, img_id='car_pose')

        if self.opt.vis:
            debugger.show_all_imgs(pause=self.pause)

