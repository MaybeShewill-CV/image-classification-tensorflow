#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/24 0:40
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV
# @File    : lr_scheduler.py
# @IDE:    PyCharm
"""
Lr scheduler
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from local_utils import config_utils


class _PolynomialDecay(object):
    """
    Polynomial decay
    """
    def __init__(self, initial_learning_rate, end_learning_rate=0.0001, power=1.0, cycle=False, name=None):
        """

        :param initial_learning_rate:
        :param end_learning_rate:
        :param power:
        :param cycle:
        :param name:
        """
        self._initial_learning_rate = initial_learning_rate
        self._end_learning_rate = end_learning_rate
        self._power = power
        self._cycle = cycle
        self._name = name

    def __call__(self, global_step, decay_steps, *args, **kwargs):
        """

        :param global_step:
        :param decay_steps:
        :param args:
        :param kwargs:
        :return:
        """
        lr = tf.train.polynomial_decay(
            learning_rate=self._initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            end_learning_rate=self._end_learning_rate,
            power=self._power,
            cycle=self._cycle,
            name=self._name
        )
        return lr


class _ExponentialDecay(object):
    """
    exponential decay
    """
    def __init__(self, initial_learning_rate, decay_rate, staircase=True, name=None):
        """

        :param initial_learning_rate:
        :param decay_rate:
        :param staircase:
        :param name:
        """
        self._initial_learning_rate = initial_learning_rate
        self._decay_rate = decay_rate
        self._staircase = staircase
        self._name = name

    def __call__(self, global_step, decay_steps, *args, **kwargs):
        """

        :param global_step:
        :param decay_steps:
        :param args:
        :param kwargs:
        :return:
        """
        lr = tf.train.exponential_decay(
            learning_rate=self._initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            decay_rate=self._decay_rate,
            staircase=self._staircase,
            name=self._name
        )
        return lr


class _CosineDecay(object):
    """
    cosine decay
    """
    def __init__(self, initial_learning_rate, alpha=0.0, name=None):
        """

        :param initial_learning_rate:
        :param alpha:
        :param name:
        """
        self._initial_learning_rate = initial_learning_rate
        self._alpha = alpha
        self._name = name

    def __call__(self, global_step, decay_steps, *args, **kwargs):
        """

        :param global_step:
        :param decay_steps:
        :param args:
        :param kwargs:
        :return:
        """
        lr = tf.train.cosine_decay(
            learning_rate=self._initial_learning_rate,
            global_step=global_step,
            decay_steps=decay_steps,
            alpha=self._alpha,
            name=self._name
        )
        return lr


class LrScheduler(object):
    """

    """
    def __init__(self, cfg):
        """

        :param cfg:
        """
        self._cfg = cfg
        self._lr_scheduler_type = self._cfg.SOLVER.LR_POLICY
        self._init_lr = self._cfg.SOLVER.LR
        if self._lr_scheduler_type.lower() == 'poly':
            self._lr_scheduler = _PolynomialDecay(
                initial_learning_rate=self._init_lr,
                end_learning_rate=self._cfg.SOLVER.POLY_DECAY.LR_POLYNOMIAL_END_LR,
                power=self._cfg.SOLVER.POLY_DECAY.LR_POLYNOMIAL_POWER,
                name='poly_decay_scheduler'
            )
        elif self._lr_scheduler_type.lower() == 'exp':
            self._lr_scheduler = _ExponentialDecay(
                initial_learning_rate=self._init_lr,
                decay_rate=self._cfg.SOLVER.EXP_DECAY.DECAY_RATE,
                staircase=self._cfg.SOLVER.EXP_DECAY.APPLY_STAIRCASE,
                name='exp_decay_scheduler'
            )
        elif self._lr_scheduler_type.lower() == 'cos':
            self._lr_scheduler = _CosineDecay(
                initial_learning_rate=self._init_lr,
                alpha=self._cfg.SOLVER.COS_DECAY.ALPHA,
                name='cos_decay_scheduler'
            )
        else:
            raise NotImplementedError('Not supported lr scheduler type of {:s}'.format(self._lr_scheduler_type))

    def __call__(self, global_step, decay_steps, *args, **kwargs):
        """

        :param global_step:
        :param decay_steps:
        :param args:
        :param kwargs:
        :return:
        """
        lr = self._lr_scheduler(
            global_step=global_step,
            decay_steps=decay_steps
        )

        return lr


def _test():
    """

    :return:
    """
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_xception.yaml')
    train_steps = 40000 * 128
    global_step = tf.Variable(tf.constant(0.0), dtype=tf.float32, trainable=False, name='global_step')
    decay_steps = tf.constant(train_steps, dtype=tf.float32, name='decay_steps')

    lr_scheduler = LrScheduler(cfg=cfg)
    global_update = tf.assign_add(global_step, tf.constant(1.0))

    with tf.control_dependencies([global_update]):
        lr = lr_scheduler(decay_steps=decay_steps, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        lrs = []
        for i in range(train_steps):
            lrs.append(sess.run(lr))
        print('Global step: {}'.format(sess.run(global_step)))
        print('Complete tf calculate')
        x = np.linspace(0, train_steps, train_steps)
        plt.plot(x, lrs)
        plt.show()

    print('Complete')


if __name__ == '__main__':
    """
    test code
    """
    _test()
