import numpy as np
from autoshape.models.losses.focal_loss import FocalLoss
from autoshape.models.losses.loss import RegP2DL1Loss, RegP3DL1Loss, BinRotLoss, PositionLoss
import paddle
import yaml
from autoshape.datasets.kitti import Kitti_dataset
from autoshape.models.autoshape import AutoShape
from autoshape.models.backbones import DLA34
from autoshape.models.heads import AutoShapePredictor
from autoshape.ops.gather import transpose_and_gather_feat
import os
import time
from autoshape.utils import TimeAverager, calculate_eta, logger
from visualdl import LogWriter
from collections import deque
import shutil

def exp_rampup(rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""

    def warpper(epoch):
        if epoch < rampup_length:
            epoch = np.clip(epoch, 0.0, rampup_length)
            phase = 1.0 - epoch / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        else:
            return 1.0

    return warpper

class BaseTrainer(object):


    def __init__(self, opt):
        self.opt = opt
        self.crit_p2d = RegP3DL1Loss()
        self.crit_rot = BinRotLoss()
        self.crit_pos = PositionLoss(opt)
        self.crit_maincenter = FocalLoss()
        self.crit_reg = RegP3DL1Loss()
        self.sigmoid = paddle.nn.Sigmoid()
        self.rampup = exp_rampup(opt.exp_rampup_epoch)

    def exp_rampup(self, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""

        def warpper(epoch):
            if epoch < rampup_length:
                epoch = np.clip(epoch, 0.0, rampup_length)
                phase = 1.0 - epoch / rampup_length
                return float(np.exp(-5.0 * phase * phase))
            else:
                return 1.0

        return warpper

    # def run_epoch(self, epoch, data_loader):
    #     self.model.train()
    #     for batch_id, batch in enumerate(data_loader()):
    #         output = self.model(batch['input'])
    #         output = output[0]
    #         # print(output['wh'].shape, batch['reg_mask'].shape, batch['ind'].shape, batch['wh'].shape)
    #         # print(output['dim'].shape, batch['reg_mask'].shape, batch['ind'].shape, batch['wh'].shape)
    #         # print(output['rot'].dtype, batch['rot_mask'].dtype, batch['ind'].dtype, batch['rotbin'].dtype, batch['rotres'].dtype)
    #         # print(batch['wh'])
    #         output['hm'] = self.sigmoid(output['hm'])
    #         loss_dict = {}
    #         loss_dict['hm_loss'] = self.crit_maincenter(output['hm'], batch['hm'])
    #         loss_dict['wh_loss'] = self.crit_reg(output['wh'], batch['wh_reg_mask'].astype(paddle.int64), batch['ind'], batch['wh'])
    #         loss_dict['dim_loss'] = self.crit_reg(output['dim'], batch['dim_reg_mask'].astype(paddle.int64), batch['ind'], batch['dim'])
    #         loss_dict['p3d_loss'] = self.crit_reg(output['p3d'], batch['p3d_reg_mask'].astype(paddle.int64), batch['ind'], batch['p3d'])
    #         loss_dict['rot_loss'] = self.crit_rot(output['rot'], batch['rot_mask'].astype(paddle.int64), batch['ind'], batch['rotbin'], batch['rotres'])
    #         loss_dict['pos_loss'] = self.crit_pos(output, batch) * self.rampup(epoch)
    #         loss_dict['hp_loss'] = self.crit_reg(output['hps'], batch['hps_mask'].astype(paddle.int64), batch['ind'], batch['hps'])
    #
    #         loss = loss_dict['hm_loss'] * self.opt.hm_weight + \
    #                loss_dict['wh_loss'] * self.opt.wh_weight + \
    #                loss_dict['dim_loss'] * self.opt.dim_weight + \
    #                loss_dict['p3d_loss'] * self.opt.dim_weight + \
    #                loss_dict['rot_loss'] * self.opt.rot_weight + \
    #                loss_dict['pos_loss'] * self.rampup(epoch) + \
    #                loss_dict['hp_loss'] * self.opt.hp_weight
    #
    #         loss.backward()
    #
    #         self.optimizer.step()
    #         self.model.clear_gradients()
    #
    #         ls = ''
    #         for k, v in loss_dict.items():
    #             ls += (k + '= {0:04f} '.format(float(v)))
    #
    #         ## test print
    #         # print()
    #         s = "[TRAIN] epoch={}, iter={}, ".format(
    #             epoch, batch_id,
    #         ) + ls
    #         print(s)

    def train(self,
              model,
              train_dataset,
              optimizer,
              ):
        """
        Launch training.

        Args:
            model（nn.Layer): A sementic segmentation model.
            train_dataset (paddle.io.Dataset): Used to read and process training datasets.
            val_dataset (paddle.io.Dataset, optional): Used to read and process validation datasets.
            optimizer (paddle.optimizer.Optimizer): The optimizer.

        """
        opt = self.opt
        max_epoch = opt.max_epoch
        batch_size = opt.batch_size
        log_iters = opt.log_iters
        save_interval = opt.save_interval
        keep_checkpoint_max = opt.keep_checkpoint_max
        save_dir = opt.save_dir

        model.train()
        nranks = paddle.distributed.ParallelEnv().nranks
        local_rank = paddle.distributed.ParallelEnv().local_rank

        start_iter = 0
        # if resume_model is not None:
        #     start_iter = resume(model, optimizer, resume_model)

        if not os.path.isdir(opt.save_dir):
            if os.path.exists(opt.save_dir):
                os.remove(opt.save_dir)
            os.makedirs(opt.save_dir)

        if nranks > 1:
            # Initialize parallel environment if not done.
            if not paddle.distributed.parallel.parallel_helper._is_parallel_ctx_initialized(
            ):
                paddle.distributed.init_parallel_env()
                ddp_model = paddle.DataParallel(model)
            else:
                ddp_model = paddle.DataParallel(model)

        batch_sampler = paddle.io.DistributedBatchSampler(
            train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True)

        loader = paddle.io.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=opt.num_workers,
            return_list=True,
        )

        # VisualDL log
        # log_writer = LogWriter(opt.save_dir)

        avg_loss = 0.0
        avg_loss_dict = {}
        iters_per_epoch = len(batch_sampler)

        reader_cost_averager = TimeAverager()
        batch_cost_averager = TimeAverager()
        save_models = deque()
        batch_start = time.time()

        iter = start_iter
        for epoch in range(max_epoch):
            iter += 1
            for batch in loader():
                iter += 1
                reader_cost_averager.record(time.time() - batch_start)

                if nranks > 1:
                    output = ddp_model(batch['input'])
                else:
                    output = model(batch['input'])

                output = output[0]
                output['hm'] = self.sigmoid(output['hm'])
                loss_dict = {}
                loss_dict['hm_loss'] = self.crit_maincenter(output['hm'], batch['hm'])
                loss_dict['wh_loss'] = self.crit_reg(output['wh'], batch['wh_reg_mask'].astype(paddle.int64),
                                                     batch['ind'], batch['wh'])
                loss_dict['dim_loss'] = self.crit_reg(output['dim'], batch['dim_reg_mask'].astype(paddle.int64),
                                                      batch['ind'], batch['dim'])
                loss_dict['p3d_loss'] = self.crit_reg(output['p3d'], batch['p3d_reg_mask'].astype(paddle.int64),
                                                      batch['ind'], batch['p3d'])
                loss_dict['rot_loss'] = self.crit_rot(output['rot'], batch['rot_mask'].astype(paddle.int64),
                                                      batch['ind'], batch['rotbin'], batch['rotres'])
                loss_dict['pos_loss'] = self.crit_pos(output, batch) * self.rampup(epoch)
                loss_dict['hp_loss'] = self.crit_reg(output['hps'], batch['hps_mask'].astype(paddle.int64),
                                                     batch['ind'], batch['hps'])

                loss = loss_dict['hm_loss'] * self.opt.hm_weight + \
                       loss_dict['wh_loss'] * self.opt.wh_weight + \
                       loss_dict['dim_loss'] * self.opt.dim_weight + \
                       loss_dict['p3d_loss'] * self.opt.dim_weight + \
                       loss_dict['rot_loss'] * self.opt.rot_weight + \
                       loss_dict['pos_loss'] * self.rampup(epoch) + \
                       loss_dict['hp_loss'] * self.opt.hp_weight
                loss.backward()

                optimizer.step()
                lr = optimizer.get_lr()
                if isinstance(optimizer._learning_rate,
                              paddle.optimizer.lr.LRScheduler):
                    optimizer._learning_rate.step()
                model.clear_gradients()
                avg_loss += loss.numpy()[0]  # get the value
                if len(avg_loss_dict) == 0:
                    avg_loss_dict = {k: v.numpy()[0] for k, v in loss_dict.items()}
                else:
                    for key, value in loss_dict.items():
                        avg_loss_dict[key] += value.numpy()[0]

                batch_cost_averager.record(
                    time.time() - batch_start, num_samples=batch_size)

                if (iter) % log_iters == 0 and local_rank == 0:
                    avg_loss /= log_iters
                    for key, value in avg_loss_dict.items():
                        avg_loss_dict[key] /= log_iters

                    remain_iters = iters_per_epoch - iter
                    avg_train_batch_cost = batch_cost_averager.get_average()
                    avg_train_reader_cost = reader_cost_averager.get_average()
                    eta = calculate_eta(remain_iters, avg_train_batch_cost)
                    ls = ''
                    for k, v in loss_dict.items():
                        ls += (k + '= {0:04f}, '.format(float(v)))
                    logger.info(
                        "[TRAIN] epoch={}, iter={}/{}, loss={:.4f}, {}, lr={:.6f}, batch_cost={:.4f}, reader_cost={:.5f} | ETA {}"
                            .format(epoch, iter, iters_per_epoch,
                                    avg_loss, ls, lr, avg_train_batch_cost,
                                    avg_train_reader_cost, eta))

                    ######################### VisualDL Log ##########################
                    # log_writer.add_scalar('Train/loss', avg_loss, iter)
                    # # Record all losses if there are more than 2 losses.
                    # for key, value in avg_loss_dict.items():
                    #     log_tag = 'Train/' + key
                    #     log_writer.add_scalar(log_tag, value, iter)
                    #
                    # log_writer.add_scalar('Train/lr', lr, iter)
                    # log_writer.add_scalar('Train/batch_cost',
                    #                       avg_train_batch_cost, iter)
                    # log_writer.add_scalar('Train/reader_cost',
                    #                       avg_train_reader_cost, iter)
                    #################################################################

                    avg_loss = 0.0
                    avg_loss_list = {}
                    reader_cost_averager.reset()
                    batch_cost_averager.reset()

                if (epoch % max_epoch == 0) and local_rank == 0:
                    current_save_dir = os.path.join(save_dir,
                                                    "epoch_{}".format(epoch))
                    if not os.path.isdir(current_save_dir):
                        os.makedirs(current_save_dir)
                    paddle.save(model.state_dict(),
                                os.path.join(current_save_dir, 'model.pdparams'))
                    paddle.save(optimizer.state_dict(),
                                os.path.join(current_save_dir, 'model.pdopt'))
                    save_models.append(current_save_dir)
                    if len(save_models) > keep_checkpoint_max > 0:
                        model_to_remove = save_models.popleft()
                        shutil.rmtree(model_to_remove)

                batch_start = time.time()

        # Sleep for half a second to let dataloader release resources.
        time.sleep(0.5)
        # log_writer.close()