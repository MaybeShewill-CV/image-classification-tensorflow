#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-26 下午4:38
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : resnet.py
# @IDE: PyCharm
"""
Resnet for image classification
"""
import time

import tensorflow as tf
import numpy as np

from cls_model_zoo import resnet_utils
from cls_model_zoo import loss
from local_utils import config_utils


class ResNet(resnet_utils.ResnetBase):
    """
    resnet model for image classification
    """
    def __init__(self, phase, cfg):
        """

        :param phase: phase of training or testing
        """
        super(ResNet, self).__init__(phase=phase)
        if phase.lower() == 'train':
            self._phase = tf.constant('train', dtype=tf.string)
        else:
            self._phase = tf.constant('test', dtype=tf.string)
        self._cfg = cfg
        self._is_training = self._init_phase()
        self._resnet_size = cfg.MODEL.RESNET.NET_SIZE
        self._block_sizes = self._get_block_sizes()
        self._block_strides = [1, 2, 2, 2]
        if self._resnet_size < 50:
            self._block_func = self._building_block_v2
        else:
            self._block_func = self._bottleneck_block_v2

        self._loss_type = self._cfg.SOLVER.LOSS_TYPE
        self._loss_func = getattr(loss, '{:s}_loss'.format(self._loss_type))
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._enable_dropout = self._cfg.TRAIN.DROPOUT.ENABLE
        if self._enable_dropout:
            self._dropout_keep_prob = self._cfg.TRAIN.DROPOUT.KEEP_PROB
        self._enable_label_smooth = self._cfg.TRAIN.LABEL_SMOOTH.ENABLE
        if self._enable_label_smooth:
            self._smooth_value = self._cfg.TRAIN.LABEL_SMOOTH.SMOOTH_VALUE
        else:
            self._smooth_value = 0.0

    def _init_phase(self):
        """
        init tensorflow bool flag
        :return:
        """
        return tf.equal(self._phase, tf.constant('train', dtype=tf.string))

    def _get_block_sizes(self):
        """

        :return:
        """
        resnet_name = 'resnet-{:d}'.format(self._resnet_size)
        block_sizes = {
            'resnet-18': [2, 2, 2, 2],
            'resnet-34': [3, 4, 6, 3],
            'resnet-50': [3, 4, 6, 3],
            'resnet-101': [3, 4, 23, 3],
            'resnet-152': [3, 8, 36, 3]
        }
        try:
            return block_sizes[resnet_name]
        except KeyError:
            raise RuntimeError('Wrong resnet name, only '
                               '[resnet-18, resnet-34, resnet-50, '
                               'resnet-101, resnet-152] supported')

    def _process_image_input_tensor(self, input_image_tensor, kernel_size,
                                    conv_stride, output_dims, pool_size,
                                    pool_stride):
        """
        Resnet entry
        :param input_image_tensor: input tensor [batch_size, h, w, c]
        :param kernel_size: kernel size
        :param conv_stride: stride of conv op
        :param output_dims: output dims of conv op
        :param pool_size: pooling window size
        :param pool_stride: pooling window stride
        :return:
        """
        inputs = self._conv2d_fixed_padding(
            inputs=input_image_tensor, kernel_size=kernel_size,
            strides=conv_stride, output_dims=output_dims, name='initial_conv_pad')
        inputs = tf.identity(inputs, 'initial_conv')

        inputs = self.maxpooling(inputdata=inputs, kernel_size=pool_size,
                                 stride=pool_stride, padding='SAME',
                                 name='initial_max_pool')

        return inputs

    def _resnet_block_layer(self, input_tensor, stride, block_nums, output_dims, name):
        """
        resnet single block.Details can be found in origin paper table 1
        :param input_tensor: input tensor [batch_size, h, w, c]
        :param stride: the conv stride in bottleneck conv_2 op
        :param block_nums: block repeat nums
        :param name: layer name
        :return:
        """
        def projection_shortcut(_inputs):
            """
            shortcut projection to align the feature maps
            :param _inputs:
            :return:
            """
            if self._resnet_size < 50:
                _output_dims = output_dims
            else:
                _output_dims = output_dims * 4

            return self._conv2d_fixed_padding(
                inputs=_inputs, output_dims=_output_dims, kernel_size=1,
                strides=stride, name='projection_shortcut')

        with tf.variable_scope(name):
            inputs = self._block_func(
                input_tensor=input_tensor,
                output_dims=output_dims,
                projection_shortcut=projection_shortcut,
                stride=stride,
                name='init_block_fn'
            )
            for index in range(1, block_nums):
                inputs = self._block_func(
                    input_tensor=inputs,
                    output_dims=output_dims,
                    projection_shortcut=None,
                    stride=1,
                    name='block_fn_{:d}'.format(index)
                )

        return inputs

    def inference(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):

            # first layer process
            inputs = self._process_image_input_tensor(
                input_image_tensor=input_tensor,
                kernel_size=7,
                conv_stride=2,
                output_dims=64,
                pool_size=3,
                pool_stride=2
            )

            # The first two layers doesn't not need apply dilation
            for index, block_nums in enumerate(self._block_sizes):
                output_dims = 64 * (2 ** index)
                inputs = self._resnet_block_layer(
                    input_tensor=inputs,
                    stride=self._block_strides[index],
                    block_nums=block_nums,
                    output_dims=output_dims,
                    name='residual_block_{:d}'.format(index + 1)
                )

            inputs = self.layerbn(
                inputdata=inputs,
                is_training=self._is_training,
                name='bn_after_block_layer'
            )
            inputs = self.relu(inputdata=inputs, name='relu_after_block_layer')

            output_tensor = self.globalavgpooling(
                inputdata=inputs,
                name='global_average_pooling'
            )
            if self._enable_dropout:
                output_tensor = tf.cond(
                    self._is_training,
                    true_fn=lambda: self.dropout(
                        inputdata=output_tensor,
                        keep_prob=self._dropout_keep_prob,
                        name='dropout_train'
                    ),
                    false_fn=lambda: tf.identity(output_tensor, name='dropout_test')
                )
            final_logits = self.fullyconnect(
                inputdata=output_tensor,
                out_dim=self._class_nums,
                use_bias=True,
                name='final_logits'
            )

        return final_logits

    def compute_loss(self, input_tensor, label, name, reuse=False):
        """

        :param input_tensor:
        :param label:
        :param name:
        :param reuse:
        :return:
        """
        logits = self.inference(
            input_tensor=input_tensor,
            name=name,
            reuse=reuse
        )

        with tf.variable_scope('resnet_loss', reuse=reuse):
            ret = self._loss_func(
                logits=logits,
                label_tensor=label,
                weight_decay=self._weights_decay,
                l2_vars=tf.trainable_variables(),
                use_label_smooth=self._enable_label_smooth,
                lb_smooth_value=self._smooth_value,
                class_nums=self._class_nums,
            )
        return ret


def get_model(phase, cfg):
    """

    :param phase:
    :param cfg:
    :return:
    """
    return ResNet(phase=phase, cfg=cfg)


def _stats_graph(graph):
    """

    :param graph:
    :return:
    """
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

    return


def _inference_time_profile():
    """

    :return:
    """
    tf.reset_default_graph()
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_resnet.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    test_label_tensor = tf.placeholder(dtype=tf.int32, shape=[1], name='test_label')
    model = get_model(phase='train', cfg=cfg)
    test_result = model.compute_loss(
        input_tensor=test_input_tensor,
        label=test_label_tensor,
        name='ResNet',
        reuse=False
    )
    tmp_logits = model.inference(input_tensor=test_input_tensor, name='ResNet', reuse=True)
    print(test_result)
    print(tmp_logits)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        t_start = time.time()
        loop_times = 5000
        for i in range(loop_times):
            _ = sess.run(tmp_logits, feed_dict={test_input_tensor: test_input})
        t_cost = time.time() - t_start
        print('Cost time: {:.5f}s'.format(t_cost / loop_times))
        print('Inference time: {:.5f} fps'.format(loop_times / t_cost))

    print('Complete')


def _model_profile():
    """

    :return:
    """
    tf.reset_default_graph()
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_resnet.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    model = get_model(phase='test', cfg=cfg)
    _ = model.inference(input_tensor=test_input_tensor, name='ResNet', reuse=False)

    with tf.Session() as sess:
        _stats_graph(sess.graph)

    print('Complete')

if __name__ == '__main__':
    """
    test code
    """
    _model_profile()

    _inference_time_profile()
