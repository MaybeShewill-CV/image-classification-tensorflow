#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/26 上午11:13
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : vgg.py
# @IDE: PyCharm
"""
vgg16 model
"""
import time

import tensorflow as tf
import numpy as np

from cls_model_zoo import cnn_basenet
from cls_model_zoo import loss
from local_utils import config_utils


class Vgg(cnn_basenet.CNNBaseModel):
    """
    vgg model for image classification
    """
    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        """
        super(Vgg, self).__init__()

        self._phase = phase
        self._cfg = cfg
        self._is_training = self._is_net_for_training()
        self._loss_type = self._cfg.SOLVER.LOSS_TYPE.lower()
        self._loss_func = getattr(loss, '{:s}_loss'.format(self._loss_type))
        self._class_nums = self._cfg.DATASET.NUM_CLASSES
        self._weights_decay = self._cfg.SOLVER.WEIGHT_DECAY
        self._vgg_net_size = self._cfg.MODEL.VGG.NET_SIZE
        self._block_size = self._get_block_sizes()
        self._block_channles = [64, 128, 256, 512, 512]
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

    def _get_block_sizes(self):
        """

        :return:
        """
        vggnet_name = 'vgg-{:d}'.format(self._vgg_net_size)
        block_sizes = {
            'vgg-16': [2, 2, 3, 3, 3],
            'vgg-19': [3, 4, 4, 4, 4],
        }
        try:
            return block_sizes[vggnet_name]
        except KeyError:
            raise RuntimeError('Wrong vgg name, only '
                               '[vgg-16, vgg-19] supported')

    def _conv_stage(self, input_tensor, k_size, out_dims, name,
                    stride=1, padding='SAME', need_layer_norm=True):
        """
        stack conv and activation in vgg
        :param input_tensor:
        :param k_size:
        :param out_dims:
        :param name:
        :param stride:
        :param padding:
        :param need_layer_norm:
        :return:
        """
        with tf.variable_scope(name):
            output = self.conv2d(
                inputdata=input_tensor, out_channel=out_dims,
                kernel_size=k_size, stride=stride,
                use_bias=True, padding=padding, name='conv'
            )

            if need_layer_norm:
                output = self.layerbn(inputdata=output, is_training=self._is_training, name='bn')

                output = self.relu(inputdata=output, name='relu')
            else:
                output = self.relu(inputdata=output, name='relu')

        return output

    def _conv_block(self, input_tensor, output_channels, conv_nums, name):
        """

        :param input_tensor:
        :param output_channels:
        :param conv_nums:
        :param name:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            output = input_tensor
            for i in range(conv_nums):
                output = self._conv_stage(
                    input_tensor=output,
                    k_size=3,
                    out_dims=output_channels,
                    stride=1,
                    padding='SAME',
                    need_layer_norm=True,
                    name='conv_stage_{:d}'.format(i + 1)
                )
            output = self.maxpooling(inputdata=output, kernel_size=2, stride=2, name='maxpooling')

        return output

    def _build_net(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            output = input_tensor
            for index, conv_nums in enumerate(self._block_size):
                block_name = 'vgg_{:d}_conv_block_{:d}'.format(self._vgg_net_size, index + 1)
                output = self._conv_block(
                    input_tensor=output,
                    output_channels=self._block_channles[index],
                    conv_nums=conv_nums,
                    name=block_name
                )
            batch_size = output.get_shape().as_list()[0]
            output_channles = output.get_shape().as_list()[-1]
            output.set_shape([batch_size, 7, 7, output_channles])
            output = self.fullyconnect(
                inputdata=output,
                out_dim=4096,
                name='fc_1'
            )
            output = self.relu(output, name='fc_1_relu')
            if self._enable_dropout:
                output = tf.cond(
                    self._is_training,
                    true_fn=lambda: self.dropout(
                        inputdata=output,
                        keep_prob=self._dropout_keep_prob,
                        name='fc_1_dropout_train'
                    ),
                    false_fn=lambda: tf.identity(output, name='fc_1_dropout_test')
                )
            output = self.fullyconnect(
                inputdata=output,
                out_dim=4096,
                name='fc_2'
            )
            output = self.relu(output, name='fc_2_relu')
            if self._enable_dropout:
                output = tf.cond(
                    self._is_training,
                    true_fn=lambda: self.dropout(
                        inputdata=output,
                        keep_prob=self._dropout_keep_prob,
                        name='fc_2_dropout_train'
                    ),
                    false_fn=lambda: tf.identity(output, name='fc_2_dropout_test')
                )
            logits = self.fullyconnect(
                inputdata=output,
                out_dim=self._class_nums,
                name='logits'
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

        with tf.variable_scope('vgg_loss', reuse=reuse):
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


def get_model(phase, cfg):
    """

    :param phase:
    :param cfg:
    :return:
    """
    return Vgg(phase=phase, cfg=cfg)


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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_vgg.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    test_label_tensor = tf.placeholder(dtype=tf.int32, shape=[1], name='test_label')
    model = get_model(phase='train', cfg=cfg)
    test_result = model.compute_loss(
        input_tensor=test_input_tensor,
        label=test_label_tensor,
        name='Vgg',
        reuse=False
    )
    tmp_logits = model.inference(input_tensor=test_input_tensor, name='Vgg', reuse=True)
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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_vgg.yaml')
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    model = get_model(phase='test', cfg=cfg)
    _ = model.inference(input_tensor=test_input_tensor, name='Vgg', reuse=False)

    with tf.Session() as sess:
        _stats_graph(sess.graph)

    print('Complete')


if __name__ == '__main__':
    """
    test code
    """
    _model_profile()

    _inference_time_profile()
