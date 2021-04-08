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
import argparse
import time
import json

import tensorflow as tf
import numpy as np
import tqdm

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
                    use_bias=True,
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

    @classmethod
    def _fuse_conv_bn_weights(cls, conv_kernel, bn_moving_avg_mean, bn_moving_avg_var, bn_gamma, bn_beta, eps=1e-7):
        """

        :param conv_kernel:
        :param bn_moving_avg_mean:
        :param bn_moving_avg_var:
        :param bn_gamma:
        :param bn_beta:
        :return:
        """
        std = np.sqrt(bn_moving_avg_var + eps)
        t = (bn_gamma / std).reshape(1, 1, 1, -1)
        output_kernel = conv_kernel * t
        output_bias = bn_beta - bn_moving_avg_mean * bn_gamma / std

        return output_kernel, output_bias

    @classmethod
    def _pad_1x1_kernel_to_3x3_kernel(cls, conv_1x1_kernel):
        """

        :param conv_1x1_kernel:
        :return:
        """
        h, w, in_channels, output_channels = conv_1x1_kernel.shape
        output = []
        for in_index in range(in_channels):
            output_c = []
            for out_index in range(output_channels):
                output_c.append(
                    np.pad(
                        conv_1x1_kernel[:, :, in_index, out_index],
                        [[1, 1], [1, 1]],
                        mode='constant',
                        constant_values=0.0
                    )
                )
            output.append(output_c)

        return np.array(output, dtype=np.float32).transpose((2, 3, 0, 1))

    @classmethod
    def _make_bn_input_layer_bn_equal_conv_kernel(cls, input_channels):
        """

        :param input_channels:
        :return:
        """
        kernel_value = np.zeros((input_channels, input_channels, 3, 3), dtype=np.float32)
        for i in range(input_channels):
            kernel_value[i, i % input_channels, 1, 1] = 1

        return kernel_value.transpose((2, 3, 0, 1))

    def _fuse_conv_block_params(self, conv_block_params):
        """

        :param conv_block_params:
        :return:
        """
        conv_1x1_sub_block = dict()
        conv_3x3_sub_block = dict()
        conv_input_sub_block = dict()

        for param_name, param_value in conv_block_params:
            if '1x1' in param_name:
                if 'moving_variance' in param_name:
                    conv_1x1_sub_block['bn_moving_avg_var'] = param_value
                elif 'moving_mean' in param_name:
                    conv_1x1_sub_block['bn_moving_avg_mean'] = param_value
                elif 'conv_1x1' in param_name:
                    conv_1x1_sub_block['conv_kernel'] = param_value
                elif 'gamma' in param_name:
                    conv_1x1_sub_block['bn_gamma'] = param_value
                elif 'beta' in param_name:
                    conv_1x1_sub_block['bn_beta'] = param_value
                else:
                    raise ValueError('Unrecognized params: {:s}'.format(param_name))
            elif '3x3' in param_name:
                if 'moving_variance' in param_name:
                    conv_3x3_sub_block['bn_moving_avg_var'] = param_value
                elif 'moving_mean' in param_name:
                    conv_3x3_sub_block['bn_moving_avg_mean'] = param_value
                elif 'conv_3x3' in param_name:
                    conv_3x3_sub_block['conv_kernel'] = param_value
                elif 'gamma' in param_name:
                    conv_3x3_sub_block['bn_gamma'] = param_value
                elif 'beta' in param_name:
                    conv_3x3_sub_block['bn_beta'] = param_value
                else:
                    raise ValueError('Unrecognized params: {:s}'.format(param_name))
            elif 'bn_input' in param_name:
                if 'moving_variance' in param_name:
                    conv_input_sub_block['bn_moving_avg_var'] = param_value
                elif 'moving_mean' in param_name:
                    conv_input_sub_block['bn_moving_avg_mean'] = param_value
                elif 'gamma' in param_name:
                    conv_input_sub_block['bn_gamma'] = param_value
                elif 'beta' in param_name:
                    conv_input_sub_block['bn_beta'] = param_value
                else:
                    raise ValueError('Unrecognized params: {:s}'.format(param_name))
            else:
                raise ValueError('Unrecognized param name: {:s}'.format(param_name))

        conv_1x1_fused_kernel, conv_1x1_fused_bias = self._fuse_conv_bn_weights(
            conv_kernel=conv_1x1_sub_block['conv_kernel'],
            bn_moving_avg_mean=conv_1x1_sub_block['bn_moving_avg_mean'],
            bn_moving_avg_var=conv_1x1_sub_block['bn_moving_avg_var'],
            bn_gamma=conv_1x1_sub_block['bn_gamma'],
            bn_beta=conv_1x1_sub_block['bn_beta']
        )
        conv_3x3_fused_kernel, conv_3x3_fused_bias = self._fuse_conv_bn_weights(
            conv_kernel=conv_3x3_sub_block['conv_kernel'],
            bn_moving_avg_mean=conv_3x3_sub_block['bn_moving_avg_mean'],
            bn_moving_avg_var=conv_3x3_sub_block['bn_moving_avg_var'],
            bn_gamma=conv_3x3_sub_block['bn_gamma'],
            bn_beta=conv_3x3_sub_block['bn_beta']
        )

        if conv_input_sub_block:
            conv_input_sub_block['conv_kernel'] = self._make_bn_input_layer_bn_equal_conv_kernel(
                input_channels=conv_input_sub_block['bn_beta'].shape[0]
            )
            conv_input_fused_kernel, conv_input_fused_bias = self._fuse_conv_bn_weights(
                conv_kernel=conv_input_sub_block['conv_kernel'],
                bn_moving_avg_mean=conv_input_sub_block['bn_moving_avg_mean'],
                bn_moving_avg_var=conv_input_sub_block['bn_moving_avg_var'],
                bn_gamma=conv_input_sub_block['bn_gamma'],
                bn_beta=conv_input_sub_block['bn_beta']
            )
            output_kernel = conv_3x3_fused_kernel + self._pad_1x1_kernel_to_3x3_kernel(conv_1x1_fused_kernel) + \
                            conv_input_fused_kernel
            output_bias = conv_3x3_fused_bias + conv_1x1_fused_bias + conv_input_fused_bias
        else:
            output_kernel = conv_3x3_fused_kernel + self._pad_1x1_kernel_to_3x3_kernel(conv_1x1_fused_kernel)
            output_bias = conv_3x3_fused_bias + conv_1x1_fused_bias

        return output_kernel, output_bias

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

        with tf.variable_scope('repvggnet_loss', reuse=reuse):
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

    def save_trained_model(self, weights_path, name, reuse=True, dump_to_json=False):
        """

        :param weights_path:
        :param name:
        :param reuse:
        :param dump_to_json:
        :return:
        """
        input_tensor = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self._cfg.AUG.EVAL_CROP_SIZE[1], self._cfg.AUG.EVAL_CROP_SIZE[0], 3],
            name='input_tensor'
        )

        _ = self._build_net(
            input_tensor=input_tensor,
            name=name,
            reuse=reuse,
            apply_reparam=False
        )

        with tf.variable_scope(name_or_scope='moving_avg'):
            variable_averages = tf.train.ExponentialMovingAverage(self._cfg.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

        trained_params = dict()
        saver = tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            saver.restore(sess=sess, save_path=weights_path)

            for vv_name, vv in variables_to_restore.items():
                trained_params[vv_name] = sess.run(vv).tolist()

        if dump_to_json:
            with open('./repvgg_trainned_params.json', 'w') as file:
                json.dump(obj=trained_params, fp=file)

        return trained_params

    def convert_reparams(self, repvgg_trainned_params: dict, repvgg_converted_param_names):
        """

        :param repvgg_trainned_params:
        :param repvgg_converted_param_names:
        :return:
        """
        converted_params = dict()
        for param_name in tqdm.tqdm(repvgg_converted_param_names):
            param_name_tmp = '/'.join(param_name.split('/')[:3]).replace(':0', '')
            output_kernel = None
            output_bias = None
            if 'final_logits' in param_name:
                if 'kernel' in param_name:
                    for trainned_param_name, trainned_param_value in repvgg_trainned_params.items():
                        if 'final_logits' in trainned_param_name and 'kernel' in trainned_param_name:
                            output_kernel = trainned_param_value
                elif 'bias' in param_name:
                    for trainned_param_name, trainned_param_value in repvgg_trainned_params.items():
                        if 'final_logits' in trainned_param_name and 'bias' in trainned_param_name:
                            output_bias = trainned_param_value
                else:
                    raise ValueError('Unrecognized params: {:s}'.format(param_name))
            else:
                corresponding_params = []
                for trainned_param_name, trainned_param_value in repvgg_trainned_params.items():
                    if 'moving_avg' in trainned_param_name:
                        trainned_param_name_tmp = '/'.join(trainned_param_name.split('/')[1:4])
                    else:
                        trainned_param_name_tmp = '/'.join(trainned_param_name.split('/')[:3])
                    if param_name_tmp == trainned_param_name_tmp:
                        corresponding_params.append((trainned_param_name, trainned_param_value))

                output_kernel, output_bias = self._fuse_conv_block_params(corresponding_params)

            if 'bias' in param_name or 'b' in param_name.split('/')[-1]:
                converted_params[param_name] = output_bias
            elif 'kernel' in param_name or 'W' in param_name.split('/')[-1] :
                converted_params[param_name] = output_kernel
            else:
                raise ValueError('Unrecognized param: {:s}'.format(param_name))

        return converted_params


def _stats_graph(graph):
    """

    :param graph:
    :return:
    """
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

    return


def get_model(phase, cfg):
    """

    :param phase:
    :param cfg:
    :return:
    """
    return RepVgg(phase=phase, cfg=cfg)


def _init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_weights_path', type=str, help='The trainned repvgg model ckpt weights path')
    parser.add_argument('--rep_params_save_path', type=str)

    return parser.parse_args()


def convert_repvgg_params():
    """

    :return:
    """
    args = _init_args()

    # init compute graph
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_repvgg.yaml')
    model = get_model(phase='train', cfg=cfg)

    # save trainned repvgg params into json
    repvgg_trainned_params = dict()
    trained_params = model.save_trained_model(weights_path=args.trained_weights_path, name='RepVgg', reuse=False)
    for param_name, param_value in trained_params.items():
        repvgg_trainned_params[param_name] = np.array(param_value, dtype=np.float32)

    # build repvgg inference compute graph
    tf.reset_default_graph()
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='test_input')
    _ = model.inference(
        input_tensor=test_input_tensor,
        name='RepVgg',
        reuse=False
    )
    repvgg_converted_names = []
    for vv in tf.trainable_variables():
        repvgg_converted_names.append(vv.name)

    converted_params = model.convert_reparams(
        repvgg_trainned_params=repvgg_trainned_params,
        repvgg_converted_param_names=repvgg_converted_names
    )

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        # train_vars = enumerate(tf.trainable_variables())

        for index, vv in enumerate(tf.trainable_variables()):
            if index == 0:
                print(vv.name)
                print(sess.run(vv))
                print('*' * 20)

        for index, vv in enumerate(tf.trainable_variables()):
            kernel = converted_params[vv.name]
            sess.run(tf.assign(vv, kernel))
            if index == 0:
                print(vv.name)
                print(sess.run(vv))
                print('*' * 20)

        saver.save(sess, save_path=args.rep_params_save_path)

    print('Complete')


def check_converted_model():
    """

    :return:
    """
    args = _init_args()

    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_repvgg.yaml')
    model = get_model(phase='train', cfg=cfg)
    test_input_ndaray = np.random.random((1, 224, 224, 3)).astype(np.float32)

    # init trained compute graph
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    logits = model._build_net(
        input_tensor=test_input_tensor,
        name='RepVgg',
        reuse=False,
        apply_reparam=False
    )

    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(cfg.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=args.trained_weights_path)

        print(sess.run(logits, feed_dict={test_input_tensor: test_input_ndaray}))
        print('*' * 100)

    # reset compute graph
    tf.reset_default_graph()
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='test_input')
    logits = model.inference(
        input_tensor=test_input_tensor,
        name='RepVgg',
        reuse=False,
    )
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=args.rep_params_save_path)

        print(sess.run(logits, feed_dict={test_input_tensor: test_input_ndaray}))
        print('*' * 100)


if __name__ == '__main__':
    """
    test code
    """
    convert_repvgg_params()

    # check_converted_model()
