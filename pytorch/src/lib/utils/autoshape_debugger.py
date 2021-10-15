from .debugger import Debugger
import math
import cv2
from .ddd_utils import compute_box_3d, project_to_image, draw_box_3d
import os

class AutoShapeDebugger(Debugger):

    def __init__(self, ipynb=False, theme='black',
               num_classes=-1, dataset=None, down_ratio=4):
        super(AutoShapeDebugger, self).__init__(ipynb=ipynb, theme=theme,
               num_classes=num_classes, dataset=dataset, down_ratio=down_ratio)
        self.kitti_cats = ['Car', 'Pedestrian', 'Cyclist']
        self.nuscenes_cats = ['car', 'pedestrian', 'bicycle', 'bus', 'construction_vehicle', 'motorcycle', 'barrier', 'traffic_cone', 'trailer', 'truck']
        self.color_table = [[247, 77, 149],
                   [32, 148, 9],
                   [166, 104, 6],
                   [7, 212, 133],
                   [1, 251, 1],
                   [2, 2, 188],
                   [219, 251, 1],
                   [96, 94, 92],
                   [229, 114, 84],
                   [216, 166, 255],
                   [113, 165, 0],
                   [8, 78, 183],
                   [112, 252, 57],
                   [5, 28, 126],
                   [100, 111, 156],
                   [140, 60, 39],
                   [75, 13, 159],
                   [188, 110, 83]
                   ]

    def save_kitti_format(self, results, img_path, opt, img_id='default', is_faster=True):

        '''
        bbox.reshape(-1, 4), dets[i, :, 4:5],  pts,   dim,   pts_score,   rot_y,   position,   prob,   cat,  p3d
        4,                   1,                n * 2, 3,     n,           1,       3,          1,      1,    n*3
        '''
        num_pt = opt.num_joints

        result_dir = opt.results_dir
        file_number = img_path.split('.')[-2][-6:]
        box = results[:4]
        if is_faster:
            score = results[4] * (1 / (1 + math.exp(-results[4 + 1 + num_pt * 2 + 3 + num_pt + 1 + 3])))
        else:
            score = (results[4] + (1 / (1 + math.exp(-results[4 + 1 + num_pt * 2 + 3 + num_pt + 1 + 3]))) + (
                        sum(results[4 + 1 + num_pt * 2 + 3:4 + 1 + num_pt * 2 + 3 + num_pt]) / num_pt)) / 3

        pos = results[5 + num_pt * 2 + 3 + num_pt + 1:5 + num_pt * 2 + 3 + num_pt + 1 + 3]
        dim = results[5 + num_pt * 2:5 + num_pt * 2 + 3]
        ori = results[5 + num_pt * 2 + 3 + num_pt]
        cat = int(results[5 + num_pt * 2 + 3 + num_pt + 1 + 3 + 1])
        det_cats = self.kitti_cats if 'kitti' in opt.dataset else self.nuscenes_cats
        self.write_detection_results(det_cats[int(cat)], result_dir, file_number, box, dim, pos, ori, score)

    def save_detection_img(self, results, img_path, opt, calib, img_id='default'):

        result_dir = opt.results_dir
        num_pt = opt.num_joints
        p2d = results[5:5 + num_pt * 2]
        pos = results[5 + num_pt * 2 + 3 + num_pt + 1:5 + num_pt * 2 + 3 + num_pt + 1 + 3]
        dim = results[5 + num_pt * 2:5 + num_pt * 2 + 3]
        ori = results[5 + num_pt * 2 + 3 + num_pt]
        cat = int(results[5 + num_pt * 2 + 3 + num_pt + 1 + 3 + 1])
        p3d = (
        results[5 + num_pt * 2 + 3 + num_pt + 1 + 3 + 1 + 1: 5 + num_pt * 2 + 3 + num_pt + 1 + 3 + 1 + 1 + num_pt * 3])
        kp_weight = results[5 + num_pt * 2 + 3 + num_pt + 1 + 3 + 1 + 1 + num_pt * 3:]
        # kp_weight = torch.sigmoid(kp_weight) / 2.0
        pos[1] = pos[1] + dim[0] / 2

        box_3d = compute_box_3d(dim, pos, ori)
        box_2d = self.project_to_image(box_3d, calib)
        self.imgs[img_id] = draw_box_3d(self.imgs[img_id], box_2d, self.color_table[cat])

        for i in range(num_pt):
            cv2.circle(self.imgs[img_id], (int(p2d[i * 2]), int(p2d[i * 2 + 1])), 3, (0, 255, 0), -1)


        result_dir = result_dir + '/res_image/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        cv2.imwrite(os.path.join(result_dir, os.path.basename(img_path)), self.imgs[img_id])