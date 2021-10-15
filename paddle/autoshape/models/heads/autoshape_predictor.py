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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from autoshape.models.layers import group_norm, sigmoid_hm
from autoshape.cvlibs import manager, param_init


@manager.HEADS.add_component
class AutoShapePredictor(nn.Layer):
    """AutoShapePredictor
    """

    def __init__(self,
                 heads_cfg,
                 num_chanels=256,
                 norm_type="gn",
                 in_channels=64):
        super().__init__()

        head_conv = num_chanels
        norm_func = nn.BatchNorm2D if norm_type == "bn" else group_norm
        self.heads = heads_cfg
        for head in heads_cfg:
            classes = heads_cfg[head]
            fc = nn.Sequential(
                nn.Conv2D(in_channels, head_conv, kernel_size=3, padding=1, bias_attr=True),
                norm_func(head_conv),
                nn.ReLU(),
                nn.Conv2D(head_conv, classes, kernel_size=1, padding=1 // 2, bias_attr=True)
            )
            if 'hm' in head:
                param_init.constant_init(fc[-1].bias, value=-2.19)
            else:
                self.init_weight(fc)
            self.__setattr__(head, fc)


    def forward(self, features):
        """predictor forward

        Args:
            features (paddle.Tensor): smoke backbone output

        Returns:
            list: dict of output heads
        """

        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(features)
        return [z]


    def init_weight(self, block):
        for sublayer in block.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                param_init.constant_init(sublayer.bias, value=0.0)


def get_channel_spec(reg_channels, name):
    """get dim and ori dim

    Args:
        reg_channels (tuple): regress channels, default(1, 2, 3, 2) for
        (depth_offset, keypoint_offset, dims, ori)
        name (str): dim or ori

    Returns:
        slice: for start channel to stop channel
    """
    if name == "dim":
        s = sum(reg_channels[:2])
        e = sum(reg_channels[:3])
    elif name == "ori":
        s = sum(reg_channels[:3])
        e = sum(reg_channels[:4])

    return slice(s, e, 1)