from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from .image import transform_preds
from .ddd_utils import ddd2locrot


def get_pred_depth(depth):
  return depth

def get_alpha(rot):
  # output: (B, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
  #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
  # return rot[:, 0]
  idx = rot[:, 1] > rot[:, 5]
  alpha1 = np.arctan(rot[:, 2] / rot[:, 3]) + (-0.5 * np.pi)
  alpha2 = np.arctan(rot[:, 6] / rot[:, 7]) + ( 0.5 * np.pi)
  return alpha1 * idx + alpha2 * (1 - idx)
  

def ddd_post_process_2d(dets, c, s, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  include_wh = dets.shape[2] > 16
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (opt.output_w, opt.output_h))
    classes = dets[i, :, -1]
    for j in range(opt.num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :3].astype(np.float32),
        get_alpha(dets[i, inds, 3:11])[:, np.newaxis].astype(np.float32),
        get_pred_depth(dets[i, inds, 11:12]).astype(np.float32),
        dets[i, inds, 12:15].astype(np.float32)], axis=1)
      if include_wh:
        top_preds[j + 1] = np.concatenate([
          top_preds[j + 1],
          transform_preds(
            dets[i, inds, 15:17], c[i], s[i], (opt.output_w, opt.output_h))
          .astype(np.float32)], axis=1)
    ret.append(top_preds)
  return ret

def ddd_post_process_3d(dets, calibs):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  ret = []
  for i in range(len(dets)):
    preds = {}
    for cls_ind in dets[i].keys():
      preds[cls_ind] = []
      for j in range(len(dets[i][cls_ind])):
        center = dets[i][cls_ind][j][:2]
        score = dets[i][cls_ind][j][2]
        alpha = dets[i][cls_ind][j][3]
        depth = dets[i][cls_ind][j][4]
        dimensions = dets[i][cls_ind][j][5:8]
        wh = dets[i][cls_ind][j][8:10]
        locations, rotation_y = ddd2locrot(
          center, alpha, dimensions, depth, calibs[0])
        bbox = [center[0] - wh[0] / 2, center[1] - wh[1] / 2,
                center[0] + wh[0] / 2, center[1] + wh[1] / 2]
        pred = [alpha] + bbox + dimensions.tolist() + \
               locations.tolist() + [rotation_y, score]
        preds[cls_ind].append(pred)
      preds[cls_ind] = np.array(preds[cls_ind], dtype=np.float32)
    ret.append(preds)
  return ret

def ddd_post_process(dets, c, s, calibs, opt):
  # dets: batch x max_dets x dim
  # return 1-based class det list
  dets = ddd_post_process_2d(dets, c, s, opt)
  dets = ddd_post_process_3d(dets, calibs)
  return dets


def ctdet_post_process(dets, c, s, h, w, num_classes):
  # dets: batch x max_dets x dim
  # return 1-based class det dict
  ret = []
  for i in range(dets.shape[0]):
    top_preds = {}
    dets[i, :, :2] = transform_preds(
          dets[i, :, 0:2], c[i], s[i], (w, h))
    dets[i, :, 2:4] = transform_preds(
          dets[i, :, 2:4], c[i], s[i], (w, h))
    classes = dets[i, :, -1]
    for j in range(num_classes):
      inds = (classes == j)
      top_preds[j + 1] = np.concatenate([
        dets[i, inds, :4].astype(np.float32),
        dets[i, inds, 4:5].astype(np.float32)], axis=1).tolist()
    ret.append(top_preds)
  return ret


def multi_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = transform_preds(dets[i, :, 5:39].reshape(-1, 2), c[i], s[i], (w, h))
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5], 
       pts.reshape(-1, 34)], axis=1).astype(np.float32).tolist()
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret
def car_pose_post_process(dets, c, s, h, w):
  # dets: batch x max_dets x 40
  # return list of 39 in image coord
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))
    pts = dets[i, :, 5:23]#transform_preds(dets[i, :, 5:23].reshape(-1, 2), c[i], s[i], (w, h))
    dim = dets[i, :, 23:26]
    #alpha =get_alpha(dets[i, :, 35:43])[:, np.newaxis].astype(np.float32)
    rot_y=dets[i,:,35:36]
    pts_score=dets[i, :, 26:35]
    prob=dets[i, :, 39:40]
    position=dets[i,:,36:39]
    cat = dets[i, :, 40:41]
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5],
       pts,pts_score,dim,rot_y,position,prob,cat], axis=1).astype(np.float32).tolist()
    #bbox score kps kps_score dim rot_y position prob
    #0:4  4:5   5:23 23:32    32:35 35:36 36:39 39:40
    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret


def autoshape_post_process(dets: object, c: object, s: object, h: object, w: object, num_pt) -> object:
  ret = []
  for i in range(dets.shape[0]):
    bbox = transform_preds(dets[i, :, :4].reshape(-1, 2), c[i], s[i], (w, h))

    pts = dets[i, :, 5: 5 + num_pt * 2]  # transform_preds(dets[i, :, 5:23].reshape(-1, 2), c[i], s[i], (w, h))
    dim = dets[i, :, 5 + num_pt * 2:5 + num_pt * 2 + 3]
    rot_y=dets[i,:,5 + num_pt * 2 + 3 + num_pt:5 + num_pt * 2 + 3 + num_pt+1]

    pts_score=dets[i, :, 5 + num_pt * 2 + 3:5 + num_pt * 2 + 3 + num_pt]
    prob=dets[i, :, 5 + num_pt * 2 + 3 + num_pt+1 + 3:5 + num_pt * 2 + 3 + num_pt+1 + 3 + 1]

    position=dets[i,:,5 + num_pt * 2 + 3 + num_pt+1:5 + num_pt * 2 + 3 + num_pt+1 + 3]

    cat = dets[i, :, 5 + num_pt * 2 + 3 + num_pt+1 + 3 + 1:5 + num_pt * 2 + 3 + num_pt+1 + 3 + 1+1]
    p3d = dets[i,:,5 + num_pt * 2 + 3 + num_pt+1 + 3 + 1+1:]
    top_preds = np.concatenate(
      [bbox.reshape(-1, 4), dets[i, :, 4:5],
       pts,dim,pts_score,rot_y,position,prob,cat, p3d], axis=1).astype(np.float32).tolist()

    ret.append({np.ones(1, dtype=np.int32)[0]: top_preds})
  return ret
