#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/30 下午5:53
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : repvgg.py
# @IDE: PyCharm
"""
repvgg model created by Xiaohan Ding "RepVGG: Making VGG-style ConvNets Great Again"
"""
import time

import tensorflow as tf
import numpy as np

from cls_model_zoo import cnn_basenet
from cls_model_zoo import loss
from local_utils import config_utils


class RepVgg(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        """
        super(RepVgg, self).__init__()

        self._phase = phase
        self._cfg = cfg

        self._is_training = self._is_net_for_training()
        self._loss_type = self._cfg.SOLVER.LOSS_TYPE.lower()
        self._loss_func = getattr(loss, '{:s}_loss'.format(self._loss_type))
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY
        self._repvgg_net_flag = self._cfg.MODEL.REPVGG.NET_FLAG
        self._repvgg_net_block_params = self._get_net_block_params()

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

    def _get_net_block_params(self):
        """

        :return:
        """
        repvggnet_name = 'repvgg-{:s}'.format(self._repvgg_net_flag)
        block_params = {
            'repvgg-a0': {
                'block_nums': [1, 2, 4, 14, 1],
                'depth_multiplier': [1.0, 0.75, 0.75, 0.75, 2.5],
                'output_channels': [64, 64, 128, 256, 512],
            },
            'repvgg-a1': {
                'block_nums': [1, 2, 4, 14, 1],
                'depth_multiplier': [1.0, 1.0, 1.0, 1.0, 2.5],
                'output_channels': [64, 64, 128, 256, 512],
            },
            'repvgg-a2': {
                'block_nums': [1, 2, 4, 14, 1],
                'depth_multiplier': [1.0, 1.5, 1.5, 1.5, 2.75],
                'output_channels': [64, 64, 128, 256, 512],
            },
            'repvgg-b0': {
                'block_nums': [1, 4, 6, 16, 1],
                'depth_multiplier': [1.0, 1.0, 1.0, 1.0, 2.5],
                'output_channels': [64, 64, 128, 256, 512],
            },
            'repvgg-b1': {
                'block_nums': [1, 4, 6, 16, 1],
                'depth_multiplier': [1.0, 2.0, 2.0, 2.0, 4.0],
                'output_channels': [64, 64, 128, 256, 512],
            },
            'repvgg-b2': {
                'block_nums': [1, 4, 6, 16, 1],
                'depth_multiplier': [1.0, 2.5, 2.5, 2.5, 5.0],
                'output_channels': [64, 64, 128, 256, 512],
            },
            'repvgg-b3': {
                'block_nums': [1, 4, 6, 16, 1],
                'depth_multiplier': [1.0, 3.0, 3.0, 3.0, 5.0],
                'output_channels': [64, 64, 128, 256, 512],
            },
        }
        try:
            return block_params[repvggnet_name]
        except KeyError:
            raise RuntimeError(
                'Wrong repvgg name, only [repvgg-a0, repvgg-a1, repvgg-a2, repvgg-b0, '
                'repvgg-b1, repvgg-b2, repvgg-b3] supported'
            )

    def _conv_block(self, input_tensor, output_channels, stride, name, padding='SAME',
                    use_bias=False, apply_reparam=False):
        """

        :param input_tensor:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param use_bias:
        :param apply_reparam:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            if apply_reparam:
                output = self.conv2d(
                    inputdata=input_tensor,
                    out_channel=output_channels,
                    kernel_size=3,
                    padding=padding,
                    stride=stride,
                    use_bias=use_bias,
                    name='conv'
                )
            else:
                input_channles = input_tensor.get_shape().as_list()[-1]
                conv_3x3 = self.conv2d(
                    inputdata=input_tensor,
                    out_channel=output_channels,
                    kernel_size=3,
                    padding=padding,
                    stride=stride,
                    use_bias=use_bias,
                    name='conv_3x3'
                )
                bn_3x3 = self.layerbn(inputdata=conv_3x3, is_training=self._is_training, name='bn_3x3', scale=True)
                conv_1x1 = self.conv2d(
                    inputdata=input_tensor,
                    out_channel=output_channels,
                    kernel_size=1,
                    padding=padding,
                    stride=stride,
                    use_bias=use_bias,
                    name='conv_1x1'
                )
                bn_1x1 = self.layerbn(inputdata=conv_1x1, is_training=self._is_training, name='bn_1x1', scale=True)

                if input_channles == output_channels and stride == 1:
                    output = tf.add_n(
                        inputs=[
                            bn_3x3,
                            bn_1x1,
                            self.layerbn(
                                inputdata=input_tensor,
                                is_training=self._is_training,
                                name='bn_input',
                                scale=True
                            ),
                        ],
                        name='add_n_fused'
                    )
                else:
                    output = tf.add_n(
                        inputs=[bn_3x3, bn_1x1],
                        name='add_n_fused'
                    )
            output = self.relu(inputdata=output, name='conv_block_output')
        return output

    def _conv_stage(self, input_tensor, output_channels, block_nums, name, apply_reparam=False):
        """

        :param input_tensor:
        :param output_channels:
        :param block_nums:
        :param name:
        :param apply_reparam:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            strides = [2] + [1] * (block_nums - 1)
            output = input_tensor
            for index, stride in enumerate(strides):
                output = self._conv_block(
                    input_tensor=output,
                    output_channels=output_channels,
                    stride=stride,
                    name='conv_block_{:d}'.format(index + 1),
                    apply_reparam=apply_reparam
                )
        output = tf.identity(output, name='{:s}_output'.format(name))
        return output

    def _build_net(self, input_tensor, name, reuse=False, apply_reparam=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :param apply_reparam:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            output = input_tensor
            for index, block_num in enumerate(self._repvgg_net_block_params['block_nums']):
                depth_multiplier = self._repvgg_net_block_params['depth_multiplier'][index]
                output_channels = int(self._repvgg_net_block_params['output_channels'][index] * depth_multiplier)
                if index == 0:
                    output_channels = min(64, output_channels)
                output = self._conv_stage(
                    input_tensor=output,
                    output_channels=output_channels,
                    block_nums=block_num,
                    name='conv_stage_{:d}'.format(index + 1),
                    apply_reparam=apply_reparam
                )
            output = self.globalavgpooling(
                inputdata=output,
                name='global_average_pooling'
            )
            logits = self.fullyconnect(
                inputdata=output,
                out_dim=self._class_nums,
                name='final_logits'
            )
        return logits

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
            reuse=reuse,
            apply_reparam=True,
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
            reuse=reuse,
            apply_reparam=False
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
    return RepVgg(phase=phase, cfg=cfg)


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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_repvgg.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='test_input')
    test_label_tensor = tf.placeholder(dtype=tf.int32, shape=[None], name='test_label')
    model = get_model(phase='train', cfg=cfg)
    output = model.compute_loss(
        input_tensor=test_input_tensor,
        label=test_label_tensor,
        name='RepVgg',
        reuse=False
    )
    # output = model.inference(input_tensor=test_input_tensor, name='RepVgg', reuse=False)
    print(output)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        _stats_graph(sess.graph)

        test_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        t_start = time.time()
        loop_times = 1000
        for i in range(loop_times):
            _ = sess.run(output, feed_dict={test_input_tensor: test_input, test_label_tensor: [1]})
        t_cost = time.time() - t_start
        print('Cost time: {:.5f}s'.format(t_cost / loop_times))
        print('Inference time: {:.5f} fps'.format(loop_times / t_cost))

    print('Complete')


if __name__ == '__main__':
    """
    test code
    """
    test()
