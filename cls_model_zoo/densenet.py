#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/25 下午1:25
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : densenet.py
# @IDE: PyCharm
"""
DenseNet model
"""
import time

import numpy as np
import tensorflow as tf

from cls_model_zoo import cnn_basenet
from cls_model_zoo import loss
from local_utils import config_utils


class DenseNet(cnn_basenet.CNNBaseModel):
    """
    densenet model for image classification
    """
    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        """
        super(DenseNet, self).__init__()
        self._cfg = cfg
        self._phase = phase
        self._is_training = self._is_net_for_training()

        self._block_nums = self._cfg.MODEL.DENSENET.BLOCK_NUMS
        self._densenet_size = cfg.MODEL.DENSENET.NET_SIZE
        self._block_depths = self._get_block_sizes()
        self._growth_rate = self._cfg.MODEL.DENSENET.GROWTH_RATE
        self._with_bc = self._cfg.MODEL.DENSENET.ENABLE_BC
        self._bc_theta = self._cfg.MODEL.DENSENET.BC_THETA

        self._loss_type = self._cfg.SOLVER.LOSS_TYPE.lower()
        self._loss_func = getattr(loss, '{:s}_loss'.format(self._loss_type))
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY

    def __str__(self):
        """

        :return:
        """
        encoder_info = 'A densenet with net size: {:d} block nums: ' \
                       '{:d} growth rate: {:d} block depth: {:d}'. \
            format(self._densenet_size, self._block_nums, self._growth_rate, self._block_depths)
        return encoder_info

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
        densenet_name = 'densenet-{:d}'.format(self._densenet_size)
        block_sizes = {
            'densenet-121': [6, 12, 24, 16],
            'densenet-169': [6, 12, 32, 32],
            'densenet-201': [6, 12, 48, 32],
            'densenet-264': [6, 12, 64, 48],
        }
        try:
            return block_sizes[densenet_name]
        except KeyError:
            raise RuntimeError('Wrong densenet name, only '
                               '[densenet-18, densenet-34, densenet-50, '
                               'densenet-101, densenet-152] supported')

    def _densenet_conv_block(self, input_tensor, k_size, output_channels, name,
                             stride=1, padding='VALID', need_bn=True, use_bias=False):
        """

        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param name:
        :param stride:
        :param padding:
        :param need_bn:
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
            if need_bn:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn', scale=False)
            result = self.relu(inputdata=result, name='densenet_conv_block_output')

            return result

    def _composite_conv(self, inputdata, out_channel, name):
        """
        Implement the composite function mentioned in DenseNet paper
        :param inputdata:
        :param out_channel:
        :param name:
        :return:
        """
        with tf.variable_scope(name):

            bn_1 = self.layerbn(inputdata=inputdata, is_training=self._is_training, name='bn_1')

            relu_1 = self.relu(bn_1, name='relu_1')

            if self._with_bc:
                conv_1 = self.conv2d(inputdata=relu_1, out_channel=out_channel,
                                     kernel_size=1,
                                     padding='SAME', stride=1, use_bias=False,
                                     name='conv_1')

                bn_2 = self.layerbn(inputdata=conv_1, is_training=self._is_training, name='bn_2')

                relu_2 = self.relu(inputdata=bn_2, name='relu_2')
                conv_2 = self.conv2d(inputdata=relu_2, out_channel=out_channel,
                                     kernel_size=3,
                                     stride=1, padding='SAME', use_bias=False,
                                     name='conv_2')
                return conv_2
            else:
                conv_2 = self.conv2d(inputdata=relu_1, out_channel=out_channel,
                                     kernel_size=3,
                                     stride=1, padding='SAME', use_bias=False,
                                     name='conv_2')
                return conv_2

    def _denseconnect_layers(self, inputdata, name):
        """
        Mainly implement the equation (2) in DenseNet paper concatenate the
        dense block feature maps
        :param inputdata:
        :param name:
        :return:
        """
        with tf.variable_scope(name):
            conv_out = self._composite_conv(
                inputdata=inputdata,
                name='composite_conv',
                out_channel=self._growth_rate
            )
            concate_cout = tf.concat(
                values=[conv_out, inputdata], axis=3, name='concatenate'
            )

        return concate_cout

    def _transition_layers(self, inputdata, name):
        """
        Mainly implement the Pooling layer mentioned in DenseNet paper
        :param inputdata:
        :param name:
        :return:
        """
        input_channels = inputdata.get_shape().as_list()[3]

        with tf.variable_scope(name):
            # First batch norm
            bn = self.layerbn(inputdata=inputdata, is_training=self._is_training, name='bn')

            # Second 1*1 conv
            if self._with_bc:
                out_channels = int(input_channels * self._bc_theta)
            else:
                out_channels = input_channels

            conv = self.conv2d(
                inputdata=bn,
                out_channel=out_channels,
                kernel_size=1,
                stride=1,
                use_bias=False,
                name='conv'
            )
            # Third average pooling
            avgpool_out = self.avgpooling(
                inputdata=conv,
                kernel_size=2,
                stride=2,
                name='avgpool'
            )
            return avgpool_out

    def _dense_block(self, inputdata, name, block_depth):
        """
        Mainly implement the dense block mentioned in DenseNet figure 1
        :param inputdata:
        :param name:
        :param block_depth:
        :return:
        """
        block_input = inputdata
        with tf.variable_scope(name):
            for i in range(block_depth):
                block_layer_name = '{:s}_layer_{:d}'.format(name, i + 1)
                block_input = self._denseconnect_layers(
                    inputdata=block_input,
                    name=block_layer_name
                )
        return block_input

    def _build_net(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            conv1 = self._densenet_conv_block(
                input_tensor=input_tensor,
                k_size=7,
                output_channels=2 * self._growth_rate,
                stride=2,
                padding='SAME',
                need_bn=True,
                use_bias=False,
                name='conv1'
            )
            max_pool1 = self.maxpooling(inputdata=conv1, kernel_size=3, stride=2, padding='SAME', name='max_pool1')
            dense_block_input = max_pool1

            # Second apply dense block stage
            for dense_block_nums in range(self._block_nums):
                dense_block_name = 'Dense_Block_{:d}'.format(dense_block_nums + 1)

                # dense connectivity
                dense_block_out = self._dense_block(
                    inputdata=dense_block_input,
                    name=dense_block_name,
                    block_depth=self._block_depths[dense_block_nums]
                )
                if dense_block_nums == self._block_nums - 1:
                    break
                # apply the trainsition part
                dense_block_out = self._transition_layers(
                    inputdata=dense_block_out,
                    name=dense_block_name
                )
                dense_block_input = dense_block_out

            result = self.globalavgpooling(
                inputdata=dense_block_out,
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

        with tf.variable_scope('densenet_loss', reuse=reuse):
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
    return DenseNet(phase=phase, cfg=cfg)


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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_densenet.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='test_input')
    test_label_tensor = tf.placeholder(dtype=tf.int32, shape=[None], name='test_label')
    model = get_model(phase='train', cfg=cfg)
    test_result = model.compute_loss(
        input_tensor=test_input_tensor,
        label=test_label_tensor,
        name='DenseNet',
        reuse=False
    )
    tmp_logits = model.inference(input_tensor=test_input_tensor, name='DenseNet', reuse=True)
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
