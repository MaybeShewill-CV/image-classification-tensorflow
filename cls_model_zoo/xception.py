#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/10 下午2:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : xception.py.py
# @IDE: PyCharm
"""
Implement xception model
"""
import collections
import time

import tensorflow as tf
import numpy as np

from cls_model_zoo import cnn_basenet
from local_utils.config_utils import parse_config_utils


class Xception(cnn_basenet.CNNBaseModel):
    """
    Xception model
    """
    def __init__(self, phase, cfg):
        """

        :param phase: whether testing or training
        """
        super(Xception, self).__init__()

        self._phase = phase
        self._cfg = cfg
        self._is_training = self._is_net_for_training()
        self._feature_maps = collections.OrderedDict()
        self._loss_type = self._cfg.SOLVER.LOSS_TYPE.lower()
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

    def _xception_conv_block(self, input_tensor, k_size, output_channels, name,
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
            result = self.relu(inputdata=result, name='xception_conv_block_output')

            return result

    def _xception_residual_conv_block(
            self, input_tensor, k_size, output_channels, name,
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
            result = self.relu(inputdata=result, name='relu')
            if need_bn:
                result = self.layerbn(inputdata=result, is_training=self._is_training, name='bn')
        return result

    def _xception_separate_conv_block(
            self, input_tensor, k_size, output_channels, name,
            stride=1, padding='SAME', need_bn=True, bn_scale=True):
        """

        :param input_tensor:
        :param k_size:
        :param output_channels:
        :param name:
        :param stride:
        :param padding:
        :param need_bn:
        :param bn_scale:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            result = self.separate_conv(
                input_tensor=input_tensor,
                output_channels=output_channels,
                kernel_size=k_size,
                name='xception_separate_conv',
                depth_multiplier=1,
                padding=padding,
                stride=stride
            )
            if need_bn:
                result = self.layerbn(
                    inputdata=result,
                    is_training=self._is_training,
                    name='xception_separate_conv_bn',
                    scale=bn_scale
                )
        return result

    def _entry_flow(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            with tf.variable_scope(name_or_scope='stage_1'):
                result = self._xception_conv_block(
                    input_tensor=input_tensor,
                    k_size=3,
                    output_channels=32,
                    name='conv_block_1',
                    stride=2,
                    use_bias=False
                )
                result = self._xception_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=64,
                    name='conv_block_2',
                    stride=1,
                    use_bias=False
                )
                residual = self._xception_residual_conv_block(
                    input_tensor=result,
                    k_size=1,
                    output_channels=128,
                    name='residual_block',
                    stride=2,
                    padding='SAME',
                    use_bias=False
                )
            with tf.variable_scope(name_or_scope='stage_2'):
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=128,
                    name='separate_conv_block_1',
                    bn_scale=False
                )
                result = self.relu(inputdata=result, name='relu_1')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=128,
                    name='separate_conv_block_2'
                )
                result = self.maxpooling(
                    inputdata=result,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    name='maxpool'
                )
                residual = tf.add(residual, result, name='residual_block_add')
                self._feature_maps['downsample_4'] = self.relu(residual, name='downsample_4_features')
                residual = self._xception_residual_conv_block(
                    input_tensor=residual,
                    k_size=1,
                    output_channels=256,
                    name='residual_block',
                    stride=2,
                    use_bias=False
                )
            with tf.variable_scope(name_or_scope='stage_3'):
                result = self.relu(result, name='relu_1')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=256,
                    name='separate_conv_block_1',
                    bn_scale=False
                )
                result = self.relu(result, name='relu_2')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=256,
                    name='separate_conv_block_2'
                )
                result = self.maxpooling(
                    inputdata=result,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    name='maxpool'
                )
                residual = tf.add(residual, result, name='residual_block_add')
                self._feature_maps['downsample_8'] = self.relu(residual, name='downsample_8_features')
                residual = self._xception_residual_conv_block(
                    input_tensor=residual,
                    k_size=1,
                    output_channels=728,
                    name='residual_block',
                    stride=2,
                    use_bias=False
                )
            with tf.variable_scope(name_or_scope='stage_4'):
                result = self.relu(result, name='relu_1')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=728,
                    name='separate_conv_block_1',
                    bn_scale=False
                )
                result = self.relu(result, name='relu_2')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=728,
                    name='separate_conv_block_2'
                )
                result = self.maxpooling(
                    inputdata=result,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    name='maxpool'
                )
                residual = tf.add(residual, result, name='residual_block_add')
        return residual

    def _middle_flow(self, input_tensor, name, repeat_times=8):
        """

        :param input_tensor:
        :param name:
        :param repeat_times
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            residual = None
            input_tensor_proj = None
            for i in range(repeat_times):
                with tf.variable_scope(name_or_scope='repeat_stack_{:d}'.format(i + 1)):
                    if residual is None:
                        residual = input_tensor
                    if input_tensor_proj is None:
                        input_tensor_proj = input_tensor
                    result = self.relu(inputdata=input_tensor_proj, name='relu_1')
                    result = self._xception_separate_conv_block(
                        input_tensor=result,
                        k_size=3,
                        output_channels=728,
                        name='separate_conv_block_1',
                        bn_scale=False
                    )
                    result = self.relu(inputdata=result, name='relu_2')
                    result = self._xception_separate_conv_block(
                        input_tensor=result,
                        k_size=3,
                        output_channels=728,
                        name='separate_conv_block_2',
                        bn_scale=False
                    )
                    result = self.relu(inputdata=result, name='relu_3')
                    result = self._xception_separate_conv_block(
                        input_tensor=result,
                        k_size=3,
                        output_channels=728,
                        name='separate_conv_block_3'
                    )
                    residual = tf.add(residual, result, name='residual_block')
                    input_tensor_proj = residual
            self._feature_maps['downsample_16'] = self.relu(residual, name='downsample_16_features')
        return residual

    def _exit_flow(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            residual = input_tensor
            with tf.variable_scope(name_or_scope='stage_1'):
                result = self.relu(inputdata=input_tensor, name='relu_1')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=728,
                    name='separate_conv_block_1',
                    bn_scale=False
                )
                result = self.relu(inputdata=result, name='relu_2')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=1024,
                    name='separate_conv_block_2'
                )
                result = self.maxpooling(
                    inputdata=result,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    name='maxpooling'
                )
                residual = self._xception_residual_conv_block(
                    input_tensor=residual,
                    k_size=1,
                    output_channels=1024,
                    name='residual_block',
                    stride=2,
                    use_bias=True
                )
                residual = tf.add(residual, result, name='residual_block_add')
            with tf.variable_scope(name_or_scope='stage_2'):
                result = self._xception_separate_conv_block(
                    input_tensor=residual,
                    k_size=3,
                    output_channels=1536,
                    name='separate_conv_block_1',
                    bn_scale=False
                )
                result = self.relu(inputdata=result, name='relu_1')
                result = self._xception_separate_conv_block(
                    input_tensor=result,
                    k_size=3,
                    output_channels=2048,
                    name='separate_conv_block_2',
                    bn_scale=False
                )
                result = self.relu(inputdata=result, name='relu_2')
                self._feature_maps['downsample_32'] = tf.identity(result)
            with tf.variable_scope(name_or_scope='stage_3'):
                self._feature_maps['global_avg_pool'] = tf.reduce_mean(result, axis=[1, 2], keepdims=True)
                result = self.globalavgpooling(
                    inputdata=result,
                    name='global_average_pooling'
                )
                result = self.fullyconnect(
                    inputdata=result,
                    out_dim=self._class_nums,
                    name='final_logits'
                )
        return result

    @property
    def feature_maps(self):
        """

        :return:
        """
        return self._feature_maps

    def _build_net(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse
        :return:
        """
        with tf.variable_scope(name, reuse=reuse):
            # firstly build entry flow
            entry_flow_output = self._entry_flow(
                input_tensor=input_tensor,
                name='entry_flow'
            )
            # secondly build middle flow
            middle_flow_output = self._middle_flow(
                input_tensor=entry_flow_output,
                name='middle_flow'
            )
            # thirdly exit flow
            exit_flow_output = self._exit_flow(
                input_tensor=middle_flow_output,
                name='exit_flow'
            )
        return exit_flow_output

    def _compute_dice_loss(self, logits, label_tensor):
        """
        dice loss is combined with bce loss here
        :param logits:
        :param label_tensor:
        :return:
        """
        def __dice_loss(_y_pred, _y_true):
            """

            :param _y_pred:
            :param _y_true:
            :return:
            """
            _intersection = tf.reduce_sum(_y_true * _y_pred, axis=-1)
            _l = tf.reduce_sum(_y_pred * _y_pred, axis=-1)
            _r = tf.reduce_sum(_y_true * _y_true, axis=-1)
            _dice = (2.0 * _intersection + 1e-5) / (_l + _r + 1e-5)
            _dice = tf.reduce_mean(_dice)
            return 1.0 - _dice

        # compute dice loss
        local_label_tensor = tf.one_hot(label_tensor, depth=self._class_nums, dtype=tf.float32)
        principal_loss_dice = __dice_loss(tf.nn.softmax(logits), local_label_tensor)
        principal_loss_dice = tf.identity(principal_loss_dice, name='principal_loss_dice')

        # compute bce loss
        principal_loss_bce = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=logits)
        )
        principal_loss_bce = tf.identity(principal_loss_bce, name='principal_loss_bce')

        # compute l2 loss
        l2_reg_loss = tf.constant(0.0, tf.float32)
        for vv in tf.trainable_variables():
            if 'beta' in vv.name or 'gamma' in vv.name:
                continue
            else:
                l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
        l2_reg_loss *= self._weights_decay
        l2_reg_loss = tf.identity(l2_reg_loss, 'l2_loss')
        total_loss = principal_loss_dice + principal_loss_bce + l2_reg_loss
        total_loss = tf.identity(total_loss, name='total_loss')

        ret = {
            'total_loss': total_loss,
            'principal_loss': principal_loss_bce + principal_loss_dice,
            'l2_loss': l2_reg_loss,
        }

        return ret

    def _compute_cross_entropy_loss(self, logits, label_tensor):
        """

        :param logits:
        :param label_tensor:
        :return:
        """
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_tensor,
            name='cross_entropy_per_example'
        )
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='cross_entropy')
        l2_reg_loss = tf.constant(0.0, tf.float32)
        for vv in tf.trainable_variables():
            if 'beta' in vv.name or 'gamma' in vv.name:
                continue
            else:
                l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
        l2_reg_loss *= self._weights_decay
        l2_reg_loss = tf.identity(l2_reg_loss, 'l2_loss')

        total_loss = cross_entropy_loss + l2_reg_loss
        total_loss = tf.identity(total_loss, 'total_loss')

        ret = {
            'total_loss': total_loss,
            'principal_loss': cross_entropy_loss,
            'l2_loss': l2_reg_loss
        }
        return ret

    @classmethod
    def _multi_category_focal_loss(cls, onehot_labels, logits, classes_weights, gamma=2.0):
        """

        :param onehot_labels:
        :param logits:
        :param classes_weights:
        :param gamma:
        :return:
        """
        epsilon = 1.e-7
        alpha = tf.multiply(onehot_labels, classes_weights)
        alpha = tf.cast(alpha, tf.float32)
        gamma = float(gamma)
        y_true = tf.cast(onehot_labels, tf.float32)
        y_pred = tf.nn.softmax(logits, dim=-1)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha)
        loss = tf.reduce_mean(fl)

        return loss

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

        with tf.variable_scope('xception_loss', reuse=reuse):
            if self._loss_type == 'cross_entropy':
                ret = self._compute_cross_entropy_loss(
                    logits=logits,
                    label_tensor=label
                )
            elif self._loss_type == 'dice':
                ret = self._compute_dice_loss(
                    logits=logits,
                    label_tensor=label
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
    return Xception(phase=phase, cfg=cfg)


def test():
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
        name='Xception',
        reuse=False
    )
    tmp_logits = model.inference(input_tensor=test_input_tensor, name='Xception', reuse=True)
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
        print('Inference time: {:.5f} fps'.format(loop_times / t_cost))

    print('Complete')


if __name__ == '__main__':
    """
    test code
    """
    test()
