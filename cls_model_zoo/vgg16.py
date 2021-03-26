#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/26 上午11:13
# @Author  : LuoYao
# @Site    : ICode
# @File    : vgg16.py
# @IDE: PyCharm
"""
vgg16 model
"""
import time

import tensorflow as tf
import numpy as np

from cls_model_zoo import cnn_basenet
from cls_model_zoo import loss
from local_utils import config_utils


class Vgg16(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        """
        super(Vgg16, self).__init__()

        self._phase = phase
        self._cfg = cfg
        self._is_training = self._is_net_for_training()
        self._loss_type = self._cfg.SOLVER.LOSS_TYPE.lower()
        self._loss_func = getattr(loss, '{:s}_loss'.format(self._loss_type))
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY

    def _is_net_for_training(self):
        """
        if the net is used for training or not
        :return:
        """
        if isinstance(self._phase, tf.Tensor):
            phase = self._phase
        else:
            phase = tf.constant(self._phase, dtype=tf.string)

        return tf.equal(phase, tf.constant('train', dtype=tf.string))


