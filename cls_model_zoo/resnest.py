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

    def _group_conv(self, input_tensor, output_channels, k_size, stride, groups, name, use_bias=False, dilations=1):
        """

        :param input_tensor:
        :param output_channels:
        :param k_size:
        :param stride:
        :param groups:
        :param name:
        :param use_bias:
        :param dilations:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            output = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding='SAME',
                stride=stride,
                split=groups,
                use_bias=use_bias,
                dilations=dilations
            )

        return output

    def _rsoftmax(self, input_tensor, filters, radix, groups, name):
        """

        :param input_tensor:
        :param filters:
        :param radix:
        :param groups:
        :param name
        :return:
        """
        x = input_tensor
        with tf.variable_scope(name_or_scope=name):
            if radix > 1:
                x = tf.reshape(x, [-1, groups, radix, filters // groups])
                x = tf.transpose(x, [0, 2, 1, 3])
                x = tf.nn.softmax(x, axis=1)
                x = tf.reshape(x, [-1, 1, 1, radix * filters])
            else:
                x = self.sigmoid(x, name='sigmoid')
            x = tf.identity(x, name='rsoftmax_output')

        return x

    def __call__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        input_tensor = kwargs['input_tensor']
        name_scope = kwargs['name']
        k_size = kwargs['k_size']
        stride = kwargs['stride']
        dilations = kwargs['dilations']
        groups = kwargs['groups']
        radix = kwargs['radix']
        output_channels = kwargs['output_channels']
        if 'padding' in kwargs:
            self._padding = kwargs['padding']
        input_channels = input_tensor.shape[-1]

        with tf.variable_scope(name_or_scope=name_scope):
            output = self._group_conv(
                input_tensor=input_tensor,
                output_channels=output_channels * radix,
                k_size=k_size,
                stride=stride,
                groups=groups,
                name='group_conv',
                use_bias=False,
                dilations=dilations
            )
            output = self.layerbn(inputdata=output, is_training=self._is_training, name='bn_1', scale=True)
            output = self.relu(output, name='relu_1')

            if radix > 1:
                splited = tf.split(output, radix, axis=-1)
                gap = sum(splited)
            else:
                gap = output

            gap = self.globalavgpooling(
                inputdata=gap,
                name='global_avg_pool',
                keepdims=True
            )
            reduction_factor = 4
            inter_channels = max(input_channels * radix // reduction_factor, 32)

            atten = self.conv2d(
                inputdata=gap,
                out_channel=inter_channels,
                kernel_size=1,
                use_bias=False,
                name='atten_conv_1'
            )
            atten = self.layerbn(
                inputdata=atten,
                is_training=self._is_training,
                name='atten_bn',
                scale=True
            )
            atten = self.relu(inputdata=atten, name='atten_relu')
            atten = self.conv2d(
                inputdata=atten,
                out_channel=output_channels,
                kernel_size=1,
                use_bias=False,
                name='atten_conv_2'
            )

            x = Conv2D(inter_channels, kernel_size=1)(gap)

            x = BatchNormalization(axis=self.channel_axis, epsilon=1.001e-5)(x)
            x = Activation(self.active)(x)
            x = Conv2D(filters * radix, kernel_size=1)(x)

            atten = self._rsoftmax(
                input_tensor=atten,
            )









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
