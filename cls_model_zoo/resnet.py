#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 19-3-26 下午4:38
# @Author  : LuoYao
# @Site    : ICode
# @File    : resnet.py
# @IDE: PyCharm
"""
Resnet for image classification
"""
import time

import tensorflow as tf
import numpy as np

from cls_model_zoo import resnet_utils
from local_utils.config_utils import parse_config_utils


class ResNet(resnet_utils.ResnetBase):
    """
    Dialated Resnet Backbone for semantic segmentation
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
            return self._conv2d_fixed_padding(
                inputs=_inputs, output_dims=output_dims * 4, kernel_size=1,
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

            inputs = tf.reduce_mean(
                input_tensor=inputs,
                axis=[1, 2],
                keepdims=True,
                name='final_reduce_mean'
            )
            inputs = tf.squeeze(input=inputs, axis=[1, 2], name='final_squeeze')

            final_logits = self.fullyconnect(
                inputdata=inputs,
                out_dim=self._cfg.DATASET.NUM_CLASSES,
                use_bias=False, name='final_logits'
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
        labels = tf.cast(label, tf.int64)

        inference_logits = self.inference(
            input_tensor=input_tensor,
            name=name,
            reuse=reuse
        )

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=inference_logits,
            labels=labels,
            name='cross_entropy_per_example'
        )
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')

        l2_loss = self._cfg.SOLVER.WEIGHT_DECAY * tf.add_n(
            [tf.nn.l2_loss(tf.cast(vv, tf.float32)) for vv in tf.trainable_variables() if 'bn' not in vv.name])

        total_loss = cross_entropy_loss + l2_loss
        total_loss = tf.identity(total_loss, name='resnet_total_loss')

        ret = {
            'total_loss': total_loss,
            'principal_loss': cross_entropy_loss,
            'l2_loss': l2_loss
        }

        return ret


def get_model(phase, cfg):
    """

    :param phase:
    :param cfg:
    :return:
    """
    return ResNet(phase=phase, cfg=cfg)


def _test():
    """

    :return:
    """
    cfg = parse_config_utils.ld_lane_xception_cfg
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 64, 256, 3], name='test_input')
    test_label_tensor = tf.placeholder(dtype=tf.int32, shape=[None], name='test_label')
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

        test_input = np.random.random((1, 64, 256, 3)).astype(np.float32)
        t_start = time.time()
        loop_times = 1000
        for i in range(loop_times):
            _ = sess.run(tmp_logits, feed_dict={test_input_tensor: test_input})
        t_cost = time.time() - t_start
        print('Cost time: {:.5f}s'.format(t_cost / loop_times))
        print('Inference time: {:.5f}fps'.format(loop_times / t_cost))

    print('Complete')


if __name__ == '__main__':
    """
    test code
    """
    _test()
