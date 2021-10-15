from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np

from models.losses import FocalLoss, RegL1Loss, RegLoss, RegWeightedL1Loss, BinRotLoss, AutoShape_Position_loss, RegWeightedL1Loss2
from models.utils import _sigmoid
from .base_trainer import BaseTrainer

class AutoShapeLoss(torch.nn.Module):
    def __init__(self, opt):
        super(AutoShapeLoss, self).__init__()
        self.crit = FocalLoss()
        self.crit_hm_hp = torch.nn.MSELoss() if opt.mse_loss else FocalLoss()
        self.crit_kp = RegWeightedL1Loss() if not opt.dense_hp else \
            torch.nn.L1Loss(reduction='sum')
        self.crit_p3d = RegWeightedL1Loss2()
        self.crit_reg = RegL1Loss() if opt.reg_loss == 'l1' else \
            RegLoss() if opt.reg_loss == 'sl1' else None
        self.crit_rot = BinRotLoss()
        self.opt = opt
        is_kitti = False
        if 'kitti' in opt.dataset: is_kitti = True
        self.position_loss = AutoShape_Position_loss(opt, is_kitti)

    def forward(self, outputs, batch, phase=None):

        opt = self.opt
        hm_loss, wh_loss, off_loss = 0, 0, 0
        hp_loss, off_loss, hm_hp_loss, hp_offset_loss = 0, 0, 0, 0
        dim_loss, rot_loss, prob_loss = 0, 0, 0
        p3d_loss = 0
        coor_loss =0
        box_score=0
        output = outputs[0]
        output['hm'] = _sigmoid(output['hm'])
        if opt.hm_hp and not opt.mse_loss:
            output['hm_hp'] = _sigmoid(output['hm_hp'])
        hm_loss = self.crit(output['hm'], batch['hm'])
        hp_loss = self.crit_kp(output['hps'],batch['hps_mask'], batch['ind'], batch['hps'],batch['dep'])
        if opt.wh_weight > 0:
            wh_loss = self.crit_reg(output['wh'], batch['reg_mask'],batch['ind'], batch['wh'])
        if opt.dim_weight > 0:
            dim_loss = self.crit_reg(output['dim'], batch['reg_mask'],batch['ind'], batch['dim'])
            p3d_loss += self.crit_reg(output['p3d'], batch['reg_mask'],
                                     batch['ind'], batch['p3d'])
        if opt.rot_weight > 0:
            rot_loss = self.crit_rot(output['rot'], batch['rot_mask'], batch['ind'], batch['rotbin'], batch['rotres'])
        if opt.reg_offset and opt.off_weight > 0:
            off_loss = self.crit_reg(output['reg'], batch['reg_mask'], batch['ind'], batch['reg'])
        if opt.reg_hp_offset and opt.off_weight > 0:
            hp_offset_loss = self.crit_reg(output['hp_offset'], batch['hp_mask'], batch['hp_ind'], batch['hp_offset'])
        if opt.hm_hp and opt.hm_hp_weight > 0:
            hm_hp_loss = self.crit_hm_hp(output['hm_hp'], batch['hm_hp'])
        coor_loss, prob_loss, box_score = self.position_loss(output, batch,phase)
        loss_stats = {'loss': box_score, 'hm_loss': hm_loss, 'hp_loss': hp_loss,
                      'hm_hp_loss': hm_hp_loss, 'hp_offset_loss': hp_offset_loss,
                      'wh_loss': wh_loss, 'off_loss': off_loss, 'dim_loss': dim_loss,
                      'rot_loss': rot_loss, 'prob_loss': prob_loss, 'box_score': box_score, 'coor_loss': coor_loss,
                      'p3d_loss': p3d_loss
                      }

        return loss_stats, loss_stats

class AutoShapeTrainer(BaseTrainer):
    def __init__(self, opt, model, optimizer=None):
        super(AutoShapeTrainer, self).__init__(opt, model, optimizer=optimizer)

    def _get_losses(self, opt):
        loss_states = ['loss', 'hm_loss', 'hp_loss', 'hm_hp_loss',
                       'hp_offset_loss', 'wh_loss', 'off_loss', 'dim_loss', 'rot_loss', 'prob_loss', 'coor_loss',
                       'box_score',
                       'p3d_loss'
                       ]
        loss = AutoShapeLoss(opt)
        return loss_states, loss

    def debug(self, batch, output, iter_id):
        pass

    def save_result(self, output, batch, results):
        pass