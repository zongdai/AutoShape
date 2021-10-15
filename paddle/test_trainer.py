import numpy as np
from autoshape.models.losses.focal_loss import FocalLoss
from autoshape.models.losses.loss import RegP2DL1Loss, RegP3DL1Loss, BinRotLoss, PositionLoss
import paddle
import yaml
from autoshape.datasets.kitti import Kitti_dataset
from autoshape.models.autoshape import AutoShape
from autoshape.models.backbones import DLA34
from autoshape.models.heads import AutoShapePredictor
from autoshape.models.trainer.base_trainer import BaseTrainer

class Opt():
    def __init__(self, cfg):
        for k, v in cfg.items():
            self.__setattr__(k, v)

if __name__ == '__main__':
    paddle.set_device("gpu")
    yaml = yaml.load(open('configs/kitti_config.yaml', 'r', encoding='utf-8').read())
    opt = Opt(yaml)
    heads_cfg = {
        'hm': opt.num_class,
        'hm_offset': 2,
        'wh': 2,
        'hps': opt.num_keypoints * 2,
        'dim': 3,
        'p3d': opt.num_keypoints * 3,
        'rot': 8,
        'kps_confidence': opt.num_keypoints * 2,
        # 'bbox_confidence': 1
    }
    backbone = DLA34(pretrained='./pretrained/dla34.pdparams')
    predictor_heads = AutoShapePredictor(heads_cfg)
    model = AutoShape(backbone, predictor_heads)

    kitti_train_dataset = Kitti_dataset(opt, 'train')
    # train_loader = paddle.io.DataLoader(kitti_train_dataset, batch_size=2, shuffle=True)

    optimizer = paddle.optimizer.Adam(learning_rate=1e-5, parameters=model.parameters())

    trainer = BaseTrainer(opt)

    trainer.train(model, kitti_train_dataset, optimizer)