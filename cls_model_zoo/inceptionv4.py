#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/5/25 下午3:57
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : inceptionv4.py
# @IDE: PyCharm
"""
InceptionV4 model
"""
import time

import numpy as np
import tensorflow as tf

from cls_model_zoo import cnn_basenet
from cls_model_zoo import loss
from local_utils import config_utils


class InceptionV4(cnn_basenet.CNNBaseModel):
    """
    Inceptionv4 model for image classification
    """

    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        """
        super(InceptionV4, self).__init__()
        self._cfg = cfg
        self._phase = phase
        self._is_training = self._is_net_for_training()

        self._reduction_a_k = self._cfg.MODEL.INCEPTION.REDUCTION_A_PARAMS.K
        self._reduction_a_l = self._cfg.MODEL.INCEPTION.REDUCTION_A_PARAMS.L
        self._reduction_a_m = self._cfg.MODEL.INCEPTION.REDUCTION_A_PARAMS.M
        self._reduction_a_n = self._cfg.MODEL.INCEPTION.REDUCTION_A_PARAMS.N

        self._loss_type = self._cfg.SOLVER.LOSS_TYPE.lower()
        self._loss_func = getattr(loss, '{:s}_loss'.format(self._loss_type))
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY
        self._enable_dropout = self._cfg.TRAIN.DROPOUT.ENABLE
        if self._enable_dropout:
            self._dropout_keep_prob = self._cfg.TRAIN.DROPOUT.KEEP_PROB
        self._enable_label_smooth = self._cfg.TRAIN.LABEL_SMOOTH.ENABLE
        if self._enable_label_smooth:
            self._smooth_value = self._cfg.TRAIN.LABEL_SMOOTH.SMOOTH_VALUE
        else:
            self._smooth_value = 0.0

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

    def _stem_block(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            # first conv stage
            output_tensor = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=32,
                stride=2,
                name='conv_1',
                padding='VALID',
                use_bias=False,
                need_activate=True
            )
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=3,
                output_channels=32,
                stride=1,
                name='conv_2',
                padding='VALID',
                use_bias=False,
                need_activate=True
            )
            output_tensor = self._conv_block(
                input_tensor=output_tensor,
                k_size=3,
                output_channels=64,
                stride=1,
                name='conv_3',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            # inception stage 1
            branch_a_output = self.maxpooling(
                inputdata=output_tensor,
                kernel_size=3,
                stride=2,
                padding='VALID',
                name='maxpool_1'
            )
            branch_b_output = self._conv_block(
                input_tensor=output_tensor,
                k_size=3,
                output_channels=96,
                stride=2,
                name='conv_4',
                padding='VALID',
                use_bias=False,
                need_activate=True
            )
            output_tensor = tf.concat([branch_a_output, branch_b_output], axis=-1, name='concat_1')

            # inception stage 2
            branch_a_output = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=64,
                stride=1,
                name='conv_5',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_a_output = self._conv_block(
                input_tensor=branch_a_output,
                k_size=3,
                output_channels=96,
                stride=1,
                name='conv_6',
                padding='VALID',
                use_bias=False,
                need_activate=True
            )
            branch_b_output = self._conv_block(
                input_tensor=output_tensor,
                k_size=1,
                output_channels=64,
                stride=1,
                name='conv_7',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_b_output = self._conv_block(
                input_tensor=branch_b_output,
                k_size=[7, 1],
                output_channels=64,
                stride=1,
                name='conv_8',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_b_output = self._conv_block(
                input_tensor=branch_b_output,
                k_size=[1, 7],
                output_channels=64,
                stride=1,
                name='conv_9',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_b_output = self._conv_block(
                input_tensor=branch_b_output,
                k_size=3,
                output_channels=96,
                stride=1,
                name='conv_10',
                padding='VALID',
                use_bias=False,
                need_activate=True
            )
            output_tensor = tf.concat([branch_a_output, branch_b_output], axis=-1, name='concat_2')

            # inception stage 3
            branch_a_output = self._conv_block(
                input_tensor=output_tensor,
                k_size=3,
                output_channels=192,
                stride=2,
                name='conv_11',
                padding='VALID',
                use_bias=False,
                need_activate=True
            )
            branch_b_output = self.maxpooling(
                inputdata=output_tensor,
                kernel_size=3,
                stride=2,
                padding='VALID',
                name='maxpool_2'
            )
            output_tensor = tf.concat([branch_a_output, branch_b_output], axis=-1, name='stem_block_output')

        return output_tensor

    def _inception_block_a(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            branch_a_output = self.avgpooling(
                inputdata=input_tensor,
                kernel_size=3,
                stride=1,
                padding='SAME',
                name='avgpool'
            )
            branch_a_output = self._conv_block(
                input_tensor=branch_a_output,
                k_size=1,
                output_channels=96,
                stride=1,
                name='branch_a_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            branch_b_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=96,
                stride=1,
                name='branch_b_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            branch_c_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=64,
                stride=1,
                name='branch_c_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_c_output = self._conv_block(
                input_tensor=branch_c_output,
                k_size=3,
                output_channels=96,
                stride=1,
                name='branch_c_conv_2',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            branch_d_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=64,
                stride=1,
                name='branch_d_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_d_output = self._conv_block(
                input_tensor=branch_d_output,
                k_size=3,
                output_channels=96,
                stride=1,
                name='branch_d_conv_2',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_d_output = self._conv_block(
                input_tensor=branch_d_output,
                k_size=3,
                output_channels=96,
                stride=1,
                name='branch_d_conv_3',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            output_tensor = tf.concat(
                [branch_a_output, branch_b_output, branch_c_output, branch_d_output],
                axis=-1,
                name='inception_block_a_output'
            )
        return output_tensor

    def _inception_block_b(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            branch_a_output = self.avgpooling(
                inputdata=input_tensor,
                kernel_size=3,
                stride=1,
                padding='SAME',
                name='avgpool'
            )
            branch_a_output = self._conv_block(
                input_tensor=branch_a_output,
                k_size=1,
                output_channels=128,
                stride=1,
                name='branch_a_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            branch_b_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=384,
                stride=1,
                name='branch_b_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            branch_c_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=192,
                stride=1,
                name='branch_c_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_c_output = self._conv_block(
                input_tensor=branch_c_output,
                k_size=[1, 7],
                output_channels=224,
                stride=1,
                name='branch_c_conv_2',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_c_output = self._conv_block(
                input_tensor=branch_c_output,
                k_size=[7, 1],
                output_channels=256,
                stride=1,
                name='branch_c_conv_3',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            branch_d_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=192,
                stride=1,
                name='branch_d_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_d_output = self._conv_block(
                input_tensor=branch_d_output,
                k_size=[1, 7],
                output_channels=192,
                stride=1,
                name='branch_d_conv_2',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_d_output = self._conv_block(
                input_tensor=branch_d_output,
                k_size=[7, 1],
                output_channels=224,
                stride=1,
                name='branch_d_conv_3',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_d_output = self._conv_block(
                input_tensor=branch_d_output,
                k_size=[1, 7],
                output_channels=224,
                stride=1,
                name='branch_d_conv_4',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_d_output = self._conv_block(
                input_tensor=branch_d_output,
                k_size=[7, 1],
                output_channels=256,
                stride=1,
                name='branch_d_conv_5',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            output_tensor = tf.concat(
                [branch_a_output, branch_b_output, branch_c_output, branch_d_output],
                axis=-1,
                name='inception_block_b_output'
            )
        return output_tensor

    def _inception_block_c(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            branch_a_output = self.avgpooling(
                inputdata=input_tensor,
                kernel_size=3,
                stride=1,
                padding='SAME',
                name='avgpool'
            )
            branch_a_output = self._conv_block(
                input_tensor=branch_a_output,
                k_size=1,
                output_channels=256,
                stride=1,
                name='branch_a_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            branch_b_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=256,
                stride=1,
                name='branch_b_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            branch_c_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=384,
                stride=1,
                name='branch_c_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_c_a_output = self._conv_block(
                input_tensor=branch_c_output,
                k_size=[1, 3],
                output_channels=256,
                stride=1,
                name='branch_c_a_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_c_b_output = self._conv_block(
                input_tensor=branch_c_output,
                k_size=[3, 1],
                output_channels=256,
                stride=1,
                name='branch_c_b_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )

            branch_d_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=384,
                stride=1,
                name='branch_d_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_d_output = self._conv_block(
                input_tensor=branch_d_output,
                k_size=[1, 3],
                output_channels=448,
                stride=1,
                name='branch_d_conv_2',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_d_output = self._conv_block(
                input_tensor=branch_d_output,
                k_size=[3, 1],
                output_channels=512,
                stride=1,
                name='branch_d_conv_3',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_d_a_output = self._conv_block(
                input_tensor=branch_d_output,
                k_size=[3, 1],
                output_channels=256,
                stride=1,
                name='branch_d_a_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_d_b_output = self._conv_block(
                input_tensor=branch_d_output,
                k_size=[1, 3],
                output_channels=256,
                stride=1,
                name='branch_d_b_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            output_tensor = tf.concat(
                [branch_a_output, branch_b_output, branch_c_a_output,
                 branch_c_b_output, branch_d_a_output, branch_d_b_output],
                axis=-1,
                name='inception_block_c_output'
            )
        return output_tensor

    def _reduction_block_a(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            branch_a_output = self.maxpooling(
                inputdata=input_tensor,
                kernel_size=3,
                stride=2,
                padding='VALID',
                name='maxpool_1'
            )

            branch_b_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=3,
                output_channels=self._reduction_a_n,
                stride=2,
                name='branch_b_conv',
                padding='VALID',
                use_bias=False,
                need_activate=False
            )

            branch_c_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=self._reduction_a_k,
                stride=1,
                name='branch_c_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_c_output = self._conv_block(
                input_tensor=branch_c_output,
                k_size=3,
                output_channels=self._reduction_a_l,
                stride=1,
                name='branch_c_conv_2',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_c_output = self._conv_block(
                input_tensor=branch_c_output,
                k_size=3,
                output_channels=self._reduction_a_m,
                stride=2,
                name='branch_c_conv_3',
                padding='VALID',
                use_bias=False,
                need_activate=True
            )

            output_tensor = tf.concat(
                [branch_a_output, branch_b_output, branch_c_output],
                axis=-1,
                name='reduction_a_output'
            )
        return output_tensor

    def _reduction_block_b(self, input_tensor, name):
        """

        :param input_tensor:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            branch_a_output = self.maxpooling(
                inputdata=input_tensor,
                kernel_size=3,
                stride=2,
                padding='VALID',
                name='maxpool_1'
            )

            branch_b_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=192,
                stride=1,
                name='branch_b_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_b_output = self._conv_block(
                input_tensor=branch_b_output,
                k_size=3,
                output_channels=192,
                stride=2,
                name='branch_b_conv_2',
                padding='VALID',
                use_bias=False,
                need_activate=True
            )

            branch_c_output = self._conv_block(
                input_tensor=input_tensor,
                k_size=1,
                output_channels=256,
                stride=1,
                name='branch_d_conv_1',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_c_output = self._conv_block(
                input_tensor=branch_c_output,
                k_size=[1, 7],
                output_channels=256,
                stride=1,
                name='branch_d_conv_2',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_c_output = self._conv_block(
                input_tensor=branch_c_output,
                k_size=[7, 1],
                output_channels=256,
                stride=1,
                name='branch_d_conv_3',
                padding='SAME',
                use_bias=False,
                need_activate=True
            )
            branch_c_output = self._conv_block(
                input_tensor=branch_c_output,
                k_size=3,
                output_channels=320,
                stride=2,
                name='branch_d_conv_4',
                padding='VALID',
                use_bias=False,
                need_activate=True
            )
            output_tensor = tf.concat(
                [branch_a_output, branch_b_output, branch_c_output],
                axis=-1,
                name='reduction_b_output'
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
            # stem block
            output_tensor = self._stem_block(
                input_tensor=input_tensor,
                name='stem'
            )
            # inception block a
            for i in range(4):
                output_tensor = self._inception_block_a(
                    input_tensor=output_tensor,
                    name='inception_block_a_{:d}'.format(i + 1)
                )
            # reduction block a
            output_tensor = self._reduction_block_a(
                input_tensor=output_tensor,
                name='recuction_block_a'
            )
            # inception block b
            for i in range(7):
                output_tensor = self._inception_block_b(
                    input_tensor=output_tensor,
                    name='inception_block_b_{:d}'.format(i + 1)
                )
            # recution block b
            output_tensor = self._reduction_block_b(
                input_tensor=output_tensor,
                name='recuction_block_b'
            )
            # inception block c
            for i in range(3):
                output_tensor = self._inception_block_c(
                    input_tensor=output_tensor,
                    name='inception_block_c_{:d}'.format(i + 1)
                )
            # cls head
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

        with tf.variable_scope('inceptionv4_loss', reuse=reuse):
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
    return InceptionV4(phase=phase, cfg=cfg)


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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_inceptionv4.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    test_label_tensor = tf.placeholder(dtype=tf.int32, shape=[1], name='test_label')
    model = get_model(phase='train', cfg=cfg)
    test_result = model.compute_loss(
        input_tensor=test_input_tensor,
        label=test_label_tensor,
        name='InceptionV4',
        reuse=False
    )
    tmp_logits = model.inference(input_tensor=test_input_tensor, name='InceptionV4', reuse=True)
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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_inceptionv4.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    model = get_model(phase='test', cfg=cfg)
    _ = model.inference(input_tensor=test_input_tensor, name='InceptionV4', reuse=False)

    with tf.Session() as sess:
        _stats_graph(sess.graph)

    print('Complete')


if __name__ == '__main__':
    """
    test code
    """
    _model_profile()

    _inference_time_profile()
