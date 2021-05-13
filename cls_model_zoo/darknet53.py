#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/5/11 下午4:38
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : darknet53.py
# @IDE: PyCharm
"""
add darknet model
"""
import time
import collections

import numpy as np
import tensorflow as tf

from cls_model_zoo import cnn_basenet
from cls_model_zoo import loss
from local_utils import config_utils


class DarkNet53(cnn_basenet.CNNBaseModel):
    """
        densenet model for image classification
        """

    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        """
        super(DarkNet53, self).__init__()
        self._cfg = cfg
        self._phase = phase
        self._is_training = self._is_net_for_training()

        self._block_params = self._get_block_params()

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

    @classmethod
    def _get_block_params(cls):
        """

        :return:
        """
        params = [
            {'output_channels': 64, 'repeat_times': 1},
            {'output_channels': 128, 'repeat_times': 2},
            {'output_channels': 256, 'repeat_times': 8},
            {'output_channels': 512, 'repeat_times': 8},
            {'output_channels': 1024, 'repeat_times': 4},
        ]

        return params

    def _conv_block(self, input_tensor, k_size, output_channels, stride,
                    name, padding='SAME', use_bias=False, need_activate=False):
        """
        conv block in attention refine
        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.conv2d(
                inputdata=input_tensor,
                out_channel=output_channels,
                kernel_size=k_size,
                padding=padding,
                stride=stride,
                use_bias=use_bias,
                name='conv'
            )
            if need_activate:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn')
                result = tf.nn.leaky_relu(features=result, name='leak_relu', alpha=0.1)
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn')
        return result

    def _residual_conv_block(self, input_tensor, output_channels,
                             name, padding='SAME', use_bias=False, need_activate=False):
        """

        :param input_tensor:
        :param output_channels:
        :param name:
        :param padding:
        :param use_bias:
        :param need_activate:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            short_cut = input_tensor

            with tf.variable_scope(name):
                residual_output = self._conv_block(
                    input_tensor=input_tensor,
                    k_size=1,
                    output_channels=int(output_channels / 2),
                    stride=1,
                    name='conv1',
                    padding=padding,
                    use_bias=use_bias,
                    need_activate=need_activate
                )
                residual_output = self._conv_block(
                    input_tensor=residual_output,
                    k_size=3,
                    output_channels=output_channels,
                    stride=1,
                    name='conv2',
                    padding=padding,
                    use_bias=use_bias,
                    need_activate=need_activate
                )
                residual_output += short_cut
                residual_output = tf.identity(residual_output, name='residual_output')

            return residual_output

    def _conv_stage(self, input_tensor, output_channels, block_repeat_times, name):
        """

        :param input_tensor:
        :param output_channels:
        :param block_repeat_times:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            # downsample input tensor
            output_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=output_channels,
                stride=2,
                name='downsample_conv',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            # apply residual block
            for i in range(block_repeat_times):
                output_tensor = self._residual_conv_block(
                    input_tensor=output_tensor,
                    output_channels=output_channels,
                    name='residual_block_{:d}'.format(i + 1),
                    need_activate=True
                )
        return output_tensor

    def _build_net(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            output_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=32,
                stride=1,
                name='conv_1',
                need_activate=True
            )

            for index, block_params in enumerate(self._block_params):
                output_channels = block_params['output_channels']
                repeat_times = block_params['repeat_times']
                output_tensor = self._conv_stage(
                    input_tensor=output_tensor,
                    output_channels=output_channels,
                    block_repeat_times=repeat_times,
                    name='conv_stage_{:d}'.format(index + 1)
                )

            result = self.globalavgpooling(
                inputdata=output_tensor,
                name='global_average_pooling'
            )
            result = self.fullyconnect(
                inputdata=result,
                out_dim=self._class_nums,
                name='final_logits'
            )

        return result

    def inference(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        logits = self._build_net(
            input_tensor=input_tensor,
            name=name,
            reuse=reuse
        )

        return logits

    def compute_loss(self, input_tensor, label, name, reuse=False):
        """

        :param input_tensor:
        :param label:
        :param name:
        :param reuse:
        :return:
        """
        logits = self._build_net(
            input_tensor=input_tensor,
            name=name,
            reuse=reuse
        )

        with tf.variable_scope('darknet_loss', reuse=reuse):
            if self._loss_type == 'cross_entropy':
                ret = self._loss_func(
                    logits=logits,
                    label_tensor=label,
                    weight_decay=self._weights_decay,
                    l2_vars=tf.trainable_variables(),
                )
            elif self._loss_type == 'dice_bce':
                ret = self._loss_func(
                    logits=logits,
                    label_tensor=label,
                    weight_decay=self._weights_decay,
                    l2_vars=tf.trainable_variables(),
                    class_nums=self._class_nums,
                )
            else:
                raise NotImplementedError('Loss of type: {:s} has not been implemented'.format(self._loss_type))

        return ret


def get_model(phase, cfg):
    """

    :param phase:
    :param cfg:
    :return:
    """
    return DarkNet53(phase=phase, cfg=cfg)


def _stats_graph(graph):
    """

    :param graph:
    :return:
    """
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

    return


def test():
    """

    :return:
    """
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_darknet53.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='test_input')
    test_label_tensor = tf.placeholder(dtype=tf.int32, shape=[None], name='test_label')
    model = get_model(phase='train', cfg=cfg)
    test_result = model.compute_loss(
        input_tensor=test_input_tensor,
        label=test_label_tensor,
        name='DarkNet53',
        reuse=False
    )
    tmp_logits = model.inference(input_tensor=test_input_tensor, name='DarkNet53', reuse=True)
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
    test()
