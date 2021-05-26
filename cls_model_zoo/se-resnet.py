#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/5/26 下午2:38
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : se-resnet.py
# @IDE: PyCharm
"""
Senet model
"""
import time

import numpy as np
import tensorflow as tf

from cls_model_zoo import resnet_utils
from cls_model_zoo import loss
from local_utils import config_utils


class SENet(resnet_utils.ResnetBase):
    """
    senet model for image classification
    """

    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        """
        super(SENet, self).__init__(phase=phase)
        self._cfg = cfg
        self._phase = phase
        self._is_training = self._is_net_for_training()

        self._se_resnet_size = cfg.MODEL.SE_RESNET.NET_SIZE
        self._block_sizes = self._get_block_sizes()
        self._block_strides = [1, 2, 2, 2]
        self._block_func = self._se_bottleneck_block
        self._se_reduction = cfg.MODEL.SE_RESNET.REDUCTION

        self._loss_type = self._cfg.SOLVER.LOSS_TYPE.lower()
        self._loss_func = getattr(loss, '{:s}_loss'.format(self._loss_type))
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY
        self._enable_dropout = self._cfg.TRAIN.DROPOUT.ENABLE
        if self._enable_dropout:
            self._dropout_keep_prob = self._cfg.TRAIN.DROPOUT.KEEP_PROB

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

    def _get_block_sizes(self):
        """

        :return:
        """
        se_resnet_name = 'se-resnet-{:d}'.format(self._se_resnet_size)
        block_sizes = {
            'se-resnet-18': [2, 2, 2, 2],
            'se-resnet-34': [3, 4, 6, 3],
            'se-resnet-50': [3, 4, 6, 3],
            'se-resnet-101': [3, 4, 23, 3],
            'se-resnet-152': [3, 8, 36, 3]
        }
        try:
            return block_sizes[se_resnet_name]
        except KeyError:
            raise RuntimeError('Wrong resnet name, only '
                               '[se-resnet-18, se-resnet-34, se-resnet-50, '
                               'se-resnet-101, se-resnet-152] supported')

    def _se_block(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        in_channles = input_tensor.get_shape().as_list()[-1]
        with tf.variable_scope(name_or_scope=name):
            output_tensor = self.globalavgpooling(
                inputdata=input_tensor,
                name='global_avg_pool',
                keepdims=True
            )
            output_tensor = self.fullyconnect(
                inputdata=output_tensor,
                out_dim=in_channles // self._se_reduction,
                use_bias=False,
                name='fc_1'
            )
            output_tensor = self.relu(
                inputdata=output_tensor,
                name='relu'
            )
            output_tensor = self.fullyconnect(
                inputdata=output_tensor,
                out_dim=in_channles,
                use_bias=False,
                name='fc_2'
            )
            output_tensor = self.sigmoid(
                inputdata=output_tensor,
                name='sigmoid'
            )
            output_tensor = tf.multiply(input_tensor, output_tensor, name='se_block_output')
        return output_tensor

    def _se_bottleneck_block(self, input_tensor, stride,
                             output_dims, projection_shortcut,
                             name):
        """
        A single bottleneck block for ResNet v2.
        Batch normalization then ReLu then convolution as described by:
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/pdf/1603.05027.pdf
        by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
        :param input_tensor: input tensor [batch, h, w, c]
        :param stride: the stride in the middle conv op
        :param output_dims: the output dims the final output dims will be output_dims * 4
        :param projection_shortcut: the project func and could be set to None if not needed
        :param name: the layer name
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            shortcut = input_tensor
            inputs = self.layerbn(inputdata=shortcut, is_training=self._is_training, name='bn_1')
            inputs = self.relu(inputdata=inputs, name='relu_1')

            if projection_shortcut is not None:
                shortcut = projection_shortcut(inputs)

            # bottleneck part1
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=1,
                output_dims=output_dims,
                strides=1,
                name='conv_pad_1'
            )
            # bottleneck part2 repalce origin conv with dilation convolution op
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_2')
            inputs = self.relu(inputdata=inputs, name='relu_2')
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=3,
                output_dims=output_dims,
                strides=stride,
                name='conv_pad_2'
            )
            # bottleneck part3
            inputs = self.layerbn(inputdata=inputs, is_training=self._is_training, name='bn_3')
            inputs = self.relu(inputdata=inputs, name='relu_3')
            inputs = self._conv2d_fixed_padding(
                inputs=inputs,
                kernel_size=1,
                output_dims=output_dims * 4,
                strides=1,
                name='conv_pad_3'
            )
            # se block
            inputs = self._se_block(input_tensor=inputs, name='se_block')

            inputs = tf.add(inputs, shortcut, name='residual_add')

        return inputs

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

    def _se_resnet_block_layer(self, input_tensor, stride, block_nums, output_dims, name):
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
            if self._se_resnet_size < 50:
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
                inputs = self._se_resnet_block_layer(
                    input_tensor=inputs,
                    stride=self._block_strides[index],
                    block_nums=block_nums,
                    output_dims=output_dims,
                    name='se_residual_block_{:d}'.format(index + 1)
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

        with tf.variable_scope('se_resnet_loss', reuse=reuse):
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
    return SENet(phase=phase, cfg=cfg)


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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_se-resnet.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    test_label_tensor = tf.placeholder(dtype=tf.int32, shape=[1], name='test_label')
    model = get_model(phase='train', cfg=cfg)
    test_result = model.compute_loss(
        input_tensor=test_input_tensor,
        label=test_label_tensor,
        name='SENet',
        reuse=False
    )
    tmp_logits = model.inference(input_tensor=test_input_tensor, name='SENet', reuse=True)
    print(test_result)
    print(tmp_logits)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        t_start = time.time()
        loop_times = 1000
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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_se-resnet.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    model = get_model(phase='test', cfg=cfg)
    _ = model.inference(input_tensor=test_input_tensor, name='SENet', reuse=False)

    with tf.Session() as sess:
        _stats_graph(sess.graph)

    print('Complete')


if __name__ == '__main__':
    """
    test code
    """
    _model_profile()

    _inference_time_profile()
