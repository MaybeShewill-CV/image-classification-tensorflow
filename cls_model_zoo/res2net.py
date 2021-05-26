#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/5/12 下午3:40
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : res2net.py
# @IDE: PyCharm
"""
Res2Net model for image classification
"""
import time
import math

import numpy as np
import tensorflow as tf

from cls_model_zoo import cnn_basenet
from cls_model_zoo import resnet_utils
from cls_model_zoo import loss
from local_utils import config_utils


class _Bottle2Neck(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        """
        super(_Bottle2Neck, self).__init__()
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
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn')
        return result

    def __call__(self, input_tensor, name, **kwargs):
        """

        :param input_tensor:
        :param kwargs:
        :return:
        """
        if 'scale' not in kwargs:
            scale = 4
        else:
            scale = kwargs['scale']
        if 'base_width' not in kwargs:
            base_width = kwargs['base_width']
        else:
            base_width = 26
        if 'projection_shortcut' not in kwargs:
            projection_shortcut = None
        else:
            projection_shortcut = kwargs['projection_shortcut']
        if 'stride' not in kwargs:
            stride = 1
        else:
            stride = kwargs['stride']
        if 'in_channels' not in kwargs:
            in_channels = input_tensor.get_shape().as_list()[-1]
        else:
            in_channels = kwargs['in_channels']

        output_channels = kwargs['output_channels']
        # in_channles = input_tensor.get_shape().as_list()[-1]
        width = int(math.floor(in_channels * (base_width / 64.0)))

        with tf.variable_scope(name_or_scope=name):
            shortcut = input_tensor
            # first 1x1 conv block
            output_tensor = self._conv_block(
                input_tensor=shortcut,
                k_size=1,
                output_channels=width * scale,
                stride=1,
                name='first_conv_1x1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            # apply residual conv block
            output_splits = tf.split(output_tensor, scale, axis=-1)
            output_tensors = []
            for index, in_tensor in enumerate(output_splits):
                if index == 0:
                    if stride != 1:
                        tmp_output_tensor = self.avgpooling(
                            inputdata=in_tensor,
                            kernel_size=3,
                            stride=stride,
                            padding='SAME',
                            name='avg_pool'
                        )
                    else:
                        tmp_output_tensor = in_tensor
                    output_tensors.append(tmp_output_tensor)
                elif index == 1:
                    tmp_output_tensor = self._conv_block(
                        input_tensor=in_tensor,
                        k_size=3,
                        output_channels=in_tensor.get_shape().as_list()[-1],
                        stride=stride,
                        name='residual_conv_{:d}'.format(index),
                        padding='SAME',
                        use_bias=False,
                        need_activate=True
                    )
                    output_tensors.append(tmp_output_tensor)
                else:
                    if stride != 1:
                        tmp_input = in_tensor
                    else:
                        tmp_input = tf.add(output_tensors[-1], in_tensor)
                    tmp_output_tensor = self._conv_block(
                        input_tensor=tmp_input,
                        k_size=3,
                        output_channels=in_tensor.get_shape().as_list()[-1],
                        stride=stride,
                        name='residual_conv_{:d}'.format(index),
                        padding='SAME',
                        use_bias=False,
                        need_activate=True
                    )
                    output_tensors.append(tmp_output_tensor)
            output_tensor = tf.concat(output_tensors, axis=-1)

            # apply final 1x1 conv
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=output_channels,
                stride=1,
                name='last_conv_1x1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            # apply residual
            if projection_shortcut is not None:
                shortcut = projection_shortcut(input_tensor)
            else:
                shortcut = input_tensor
            output_tensor = tf.add(shortcut, output_tensor, name='residual_add')

        return output_tensor

    def inference(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        pass

    def compute_loss(self, input_tensor, label, name, reuse=False):
        """

        :param input_tensor:
        :param label:
        :param name:
        :param reuse:
        :return:
        """
        pass


class Res2Net(resnet_utils.ResnetBase):
    """

    """
    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        """
        super(Res2Net, self).__init__(phase=phase)
        self._phase = phase
        self._cfg = cfg
        self._is_training = self._is_net_for_training()

        # set res2net params
        self._res2net_size = cfg.MODEL.RES2NET.NET_SIZE
        self._block_sizes = self._get_block_sizes()
        self._bottleneck_scale = cfg.MODEL.RES2NET.BOTTLENECK_SCALE
        self._bottleneck_base_width = cfg.MODEL.RES2NET.BOTTLENECK_BASE_WIDTH
        self._block_strides = [1, 2, 2, 2]
        self._block_func = _Bottle2Neck(cfg=self._cfg, phase=phase)

        # set model hyper params
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY
        self._loss_type = self._cfg.SOLVER.LOSS_TYPE
        self._loss_func = getattr(loss, '{:s}_loss'.format(self._loss_type))
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
        resnet_name = 'res2net-{:d}'.format(self._res2net_size)
        block_sizes = {
            'res2net-18': [2, 2, 2, 2],
            'res2net-50': [3, 4, 6, 3],
            'res2net-101': [3, 4, 23, 3],
        }
        try:
            return block_sizes[resnet_name]
        except KeyError:
            raise RuntimeError('Wrong res2net name, only [res2net-18, res2net-50, res2net-101] supported')

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
                result = self.relu(inputdata=result, name='relu')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn')
        return result

    def _res2net_block_layer(self, input_tensor, stride, block_nums, output_dims, name):
        """
        resnet single block.Details can be found in origin paper table 1
        :param input_tensor: input tensor [batch_size, h, w, c]
        :param stride: the conv stride in bottleneck conv_2 op
        :param block_nums: block repeat nums
        :param name: layer name
        :return:
        """
        if self._res2net_size < 50:
            _output_dims = output_dims
        else:
            _output_dims = output_dims * 4
        _in_channles = output_dims

        def projection_shortcut(_inputs):
            """
            shortcut projection to align the feature maps
            :param _inputs:
            :return:
            """
            return self._conv2d_fixed_padding(
                inputs=_inputs, output_dims=_output_dims, kernel_size=1,
                strides=stride, name='projection_shortcut')

        with tf.variable_scope(name):
            inputs = self._block_func(
                input_tensor=input_tensor,
                output_channels=_output_dims,
                projection_shortcut=projection_shortcut,
                scale=self._bottleneck_scale,
                base_width=self._bottleneck_base_width,
                name='init_block_fn',
                in_channels=_in_channles,
                stride=stride
            )
            for index in range(1, block_nums):
                inputs = self._block_func(
                    input_tensor=inputs,
                    output_channels=_output_dims,
                    projection_shortcut=None,
                    scale=self._bottleneck_scale,
                    base_width=self._bottleneck_base_width,
                    name='block_fn_{:d}'.format(index),
                    in_channels=_in_channles,
                    stride=1
                )

        return inputs

    def _build_net(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # apply first conv layer
            ouptut_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=7,
                output_channels=64,
                stride=2,
                name='conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            ouptut_tensor = self.maxpooling(
                inputdata=ouptut_tensor, kernel_size=3, stride=2, padding='SAME', name='maxpool_1'
            )

            # apply residual block
            for index, block_nums in enumerate(self._block_sizes):
                output_dims = 64 * (2 ** index)
                ouptut_tensor = self._res2net_block_layer(
                    input_tensor=ouptut_tensor,
                    stride=self._block_strides[index],
                    block_nums=block_nums,
                    output_dims=output_dims,
                    name='res2net_bottleneck_{:d}'.format(index + 1)
                )

            # apply global avg
            output_tensor = self.globalavgpooling(
                inputdata=ouptut_tensor,
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

        with tf.variable_scope('res2net_loss', reuse=reuse):
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
    return Res2Net(phase=phase, cfg=cfg)


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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_res2net.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    test_label_tensor = tf.placeholder(dtype=tf.int32, shape=[1], name='test_label')
    model = get_model(phase='train', cfg=cfg)
    test_result = model.compute_loss(
        input_tensor=test_input_tensor,
        label=test_label_tensor,
        name='Res2Net',
        reuse=False
    )
    tmp_logits = model.inference(input_tensor=test_input_tensor, name='Res2Net', reuse=True)
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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_res2net.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    model = get_model(phase='train', cfg=cfg)
    _ = model.inference(input_tensor=test_input_tensor, name='Res2Net', reuse=False)

    with tf.Session() as sess:
        _stats_graph(sess.graph)

    print('Complete')


if __name__ == '__main__':
    """
    test code
    """
    _model_profile()

    _inference_time_profile()
