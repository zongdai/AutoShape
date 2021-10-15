from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .car_pose import CarPoseTrainer
from .autoshape import AutoShapeTrainer

train_factory = {
  'car_pose': CarPoseTrainer,
  'auto_shape': AutoShapeTrainer
}
