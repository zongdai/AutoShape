from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import _init_paths

import os
from detectors.auto_shape import AutoShapeDetector
from opts import opts
import shutil
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['net', 'dec']


def read_calib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 2:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 4)
            return calib

def demo(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    opt.heads = {'hm': 1, 'wh': 2, 'hps': opt.num_joints*2, 'rot': 8, 'dim': 3, 'p3d': opt.num_joints*3, 'adaptive_weights': opt.num_joints*2, 'prob': 1}
    opt.hm_hp=False
    opt.reg_offset=False
    opt.reg_hp_offset=False
    opt.faster=True
    Detector = AutoShapeDetector
    detector = Detector(opt)
    if os.path.exists(opt.results_dir):
        shutil.rmtree(opt.results_dir,True)

    image_names = []
    calibs = []

    if os.path.isdir(opt.demo):
      ls = os.listdir(opt.demo)
      for file_name in sorted(ls):
          ext = file_name[file_name.rfind('.') + 1:].lower()
          if ext in image_ext:
              image_names.append(os.path.join(opt.demo, file_name))
              calibs.append(read_calib(os.path.join(opt.calib_dir, file_name.split('.')[0] + '.txt')))
    else:
          if opt.demo[-3:] == 'txt':
              with open(opt.demo, 'r') as f:
                  lines = f.readlines()
              image_names = [os.path.join(opt.demo, img[:6] + '.png') for img in lines]
              calibs = [read_calib(os.path.join(opt.calib_dir, img[:6] + '.txt')) for img in lines]
          else:
              image_names = [opt.demo]
              calibs = [read_calib(os.path.join(opt.calib_dir, opt.demo.split('.')[0] + '.txt'))]
    time_tol=0
    num=0
    for (image_name, calib) in zip(image_names, calibs):
      num+=1
      ret = detector.run(image_name, calib)
      time_str = ''
      for stat in time_stats:
          time_tol=time_tol+ret[stat]
          time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      time_str=time_str+'{} {:.3f}s |'.format('tol', time_tol/num)
      print(time_str)
if __name__ == '__main__':
    opt = opts().init()
    demo(opt)
    # --demo  /home/beta/SKP/data/Kitti3D/test/image/ --calib_dir /home/beta/SKP/data/Kitti3D/test/calib_test/calib --load_model /home/beta/Baidu_Projects/baidu/personal-code/Densekeypoint_Centernet/kitti_format/trainval_model_batch16_gpu2_200epoch_pc48_adaptivekp_rightaug.pth --gpus 0 --arch dla_34