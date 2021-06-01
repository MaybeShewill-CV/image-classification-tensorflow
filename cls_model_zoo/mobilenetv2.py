#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/6/22 下午3:54
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : mobilenetv2.py
# @IDE: PyCharm
"""
MobileNetV2 tensorflow implementation
"""
import collections
import time

import numpy as np
import tensorflow as tf

from cls_model_zoo import cnn_basenet
from cls_model_zoo import loss
from local_utils import config_utils


class MobileNetV2(cnn_basenet.CNNBaseModel):
    """
    mobilenetV2 model for image classification
    """
    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        """
        super(MobileNetV2, self).__init__()
        self._phase = phase
        self._cfg = cfg
        self._is_training = self._is_net_for_training()

        # set model hyper params
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY
        self._loss_type = self._cfg.SOLVER.LOSS_TYPE
        self._loss_func = getattr(loss, '{:s}_loss'.format(self._loss_type))
        self._enable_ohem = self._cfg.SOLVER.OHEM.ENABLE
        if self._enable_ohem:
            self._ohem_score_thresh = self._cfg.SOLVER.OHEM.SCORE_THRESH
            self._ohem_min_sample_nums = self._cfg.SOLVER.OHEM.MIN_SAMPLE_NUMS
        self._enable_dropout = self._cfg.TRAIN.DROPOUT.ENABLE
        if self._enable_dropout:
            self._dropout_keep_prob = self._cfg.TRAIN.DROPOUT.KEEP_PROB
        self._enable_label_smooth = self._cfg.TRAIN.LABEL_SMOOTH.ENABLE
        if self._enable_label_smooth:
            self._smooth_value = self._cfg.TRAIN.LABEL_SMOOTH.SMOOTH_VALUE
        else:
            self._smooth_value = 0.0

        # build bottleneck hyper params
        self._bottleneck_hyper_params = self._build_bottleneck_layers_hyper_params()

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
    def _build_bottleneck_layers_hyper_params(cls):
        """

        :return:
        """
        params = [
            ('bottleneck_stage_1', (1, 16, 1, 1)),
            ('bottleneck_stage_2', (6, 24, 2, 2)),
            ('bottleneck_stage_3', (6, 32, 3, 2)),
            ('bottleneck_stage_4', (6, 64, 4, 2)),
            ('bottleneck_stage_5', (6, 96, 3, 1)),
            ('bottleneck_stage_6', (6, 160, 3, 2)),
            ('bottleneck_stage_7', (6, 320, 1, 1)),
        ]
        return collections.OrderedDict(params)

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
                result = tf.nn.relu6(features=result, name='relu6')
            else:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn')
        return result

    def _bottleneck_layer(self, input_tensor, t, output_channels, stride, name, padding='SAME', with_residuals=True):
        """

        :param input_tensor:
        :param t:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param with_residuals:
        :return:
        """
        [_, _, _, in_c] = input_tensor.get_shape().as_list()
        with tf.variable_scope(name_or_scope=name):
            output_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=int(t * in_c),
                stride=1,
                name='conv_1x1',
                padding=padding,
                use_bias=False,
                need_activate=True
            )
            output_tensor = self.depthwise_conv(
                input_tensor=output_tensor,
                kernel_size=3,
                stride=stride,
                padding=padding,
                name='conv_3x3_depthwise'
            )
            output_tensor = self.layerbn(
                inputdata=output_tensor,
                is_training=self._is_training,
                name='conv_3x3_depthwise_bn'
            )
            output_tensor = tf.nn.relu6(features=output_tensor, name='conv_3x3_depthwise_relu6')
            output_tensor = self.conv2d(
                inputdata=output_tensor,
                out_channel=output_channels,
                kernel_size=1,
                padding=padding,
                stride=1,
                use_bias=False,
                name='conv_1x1_output'
            )
            if with_residuals:
                output_tensor = tf.add(input_tensor, output_tensor, name='bottleneck_output')
            else:
                output_tensor = tf.identity(output_tensor, name='bottleneck_output')
        return output_tensor

    def _build_net(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            # first conv layer
            output_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=32,
                stride=2,
                name='first_conv_3x3',
                use_bias=False,
                need_activate=True
            )
            # bottleneck stages
            for stage_name, stage_params in self._bottleneck_hyper_params.items():
                t = stage_params[0]
                output_channels = stage_params[1]
                repeat_times = stage_params[2]
                for repeat_index in range(repeat_times):
                    if repeat_index == 0:
                        stride = stage_params[3]
                        with_residuals = False
                    else:
                        stride = 1
                        with_residuals = True
                    output_tensor = self._bottleneck_layer(
                        input_tensor=output_tensor,
                        t=t,
                        output_channels=output_channels,
                        stride=stride,
                        name='{:s}_block_{:d}'.format(stage_name, repeat_index + 1),
                        with_residuals=with_residuals
                    )
            # feature mapping
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=1280,
                stride=1,
                name='feature_map_conv_1x1',
                use_bias=False,
                need_activate=True
            )
            output_tensor = self.globalavgpooling(
                inputdata=output_tensor,
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
            output_tensor = self.fullyconnect(
                inputdata=output_tensor,
                out_dim=self._class_nums,
                name='final_logits'
            )
        return output_tensor

    def inference(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            logits = self._build_net(
                input_tensor=input_tensor,
                name='inference',
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
        with tf.variable_scope(name_or_scope=name):
            logits = self._build_net(
                input_tensor=input_tensor,
                name='inference',
                reuse=reuse
            )

            with tf.variable_scope('mobilenetv2_loss', reuse=reuse):
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
    return MobileNetV2(phase=phase, cfg=cfg)


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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_mobilenetv2.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    test_label_tensor = tf.placeholder(dtype=tf.int32, shape=[1], name='test_label')
    model = get_model(phase='train', cfg=cfg)
    test_result = model.compute_loss(
        input_tensor=test_input_tensor,
        label=test_label_tensor,
        name='MobileNetV2',
        reuse=False
    )
    tmp_logits = model.inference(input_tensor=test_input_tensor, name='MobileNetV2', reuse=True)
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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_mobilenetv2.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    model = get_model(phase='test', cfg=cfg)
    _ = model.inference(input_tensor=test_input_tensor, name='MobileNetV2', reuse=False)

    with tf.Session() as sess:
        _stats_graph(sess.graph)

    print('Complete')


if __name__ == '__main__':
    """
    test code
    """
    _model_profile()

    _inference_time_profile()
