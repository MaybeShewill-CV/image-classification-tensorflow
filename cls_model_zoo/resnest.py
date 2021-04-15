#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/16 0:38
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV
# @File    : resnest.py
# @IDE:    PyCharm
"""
ResNeSt model for image classification
"""
import time

import tensorflow as tf
import numpy as np

from cls_model_zoo import cnn_basenet
from cls_model_zoo import loss
from local_utils import config_utils


class _SplitGroupConv(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, phase):
        """

        :param phase:
        """
        super(_SplitGroupConv, self).__init__()
        self._phase = phase
        self._is_training = self._is_net_for_training()
        self._padding = 'SAME'

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

    def compute_loss(self, input_tensor, label, name, reuse=False):
        """

        :param input_tensor:
        :param label:
        :param name:
        :param reuse:
        :return:
        """
        pass

    def inference(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        pass


class ResNeSt(cnn_basenet.CNNBaseModel):
    """
    ResNeSt model for image classification
    """
    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        """
        super(ResNeSt, self).__init__()

        self._phase = phase
        self._cfg = cfg
        self._is_training = self._is_net_for_training()

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

    def compute_loss(self, input_tensor, label, name, reuse=False):
        """

        :param input_tensor:
        :param label:
        :param name:
        :param reuse:
        :return:
        """
        pass

    def inference(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        pass


def get_model(phase, cfg):
    """

    :param phase:
    :param cfg:
    :return:
    """
    return ResNeSt(phase=phase, cfg=cfg)


def _stats_graph(graph):
    """

    :param graph:
    :return:
    """
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

    return


def _test():
    """

    :return:
    """
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_resnest.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='test_input')
    test_label_tensor = tf.placeholder(dtype=tf.int32, shape=[None], name='test_label')
    model = get_model(phase='train', cfg=cfg)
    test_result = model.compute_loss(
        input_tensor=test_input_tensor,
        label=test_label_tensor,
        name='ResNeSt',
        reuse=False
    )
    tmp_logits = model.inference(input_tensor=test_input_tensor, name='ResNeSt', reuse=True)
    print(test_result)
    print(tmp_logits)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        _stats_graph(sess.graph)

        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        t_start = time.time()
        loop_times = 1000
        for i in range(loop_times):
            _ = sess.run(tmp_logits, feed_dict={test_input_tensor: test_input})
        t_cost = time.time() - t_start
        print('Cost time: {:.5f}s'.format(t_cost / loop_times))
        print('Inference time: {:.5f} fps'.format(loop_times / t_cost))

    print('Complete')


if __name__ == '__main__':
    """
    test code
    """
    _test()
