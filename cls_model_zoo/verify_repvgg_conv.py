#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/9 下午2:31
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : verify_repvgg_conv.py
# @IDE: PyCharm
"""

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

PADDING = 'VALID'
PHASE = 'test'
INPUT_TENSOR_SIZE = [224, 224]
INPUT_CHANNELS = 3
OUTPUT_CHANNELS = 64
STRIDE = 2


class RepVggTest(cnn_basenet.CNNBaseModel):
    """

    """
    def __init__(self, phase, cfg):
        """

        :param phase:
        :param cfg:
        """
        super(RepVggTest, self).__init__()
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

        # return np.array(output, dtype=np.float32).reshape((3, 3, in_channels, output_channels))
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
        # return kernel_value.reshape((3, 3, input_channels, input_channels))

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

    def conv_block(self, input_tensor, output_channels, stride, name, padding='VALID', apply_reparam=False):
        """

        :param input_tensor:
        :param output_channels:
        :param stride:
        :param name:
        :param padding:
        :param apply_reparam:
        :return:
        """
        with tf.variable_scope(name_or_scope=name):
            if apply_reparam:
                if padding == 'VALID':
                    input_tensor = tf.keras.layers.ZeroPadding2D().call(inputs=input_tensor)
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
                conv_3x3_input_tensor = tf.keras.layers.ZeroPadding2D().call(inputs=input_tensor) if \
                    padding == 'VALID' else input_tensor
                conv_3x3 = self.conv2d(
                    inputdata=conv_3x3_input_tensor,
                    out_channel=output_channels,
                    kernel_size=3,
                    padding=padding,
                    stride=stride,
                    use_bias=False,
                    name='conv_3x3'
                )
                bn_3x3 = self.layerbn(inputdata=conv_3x3, is_training=self._is_training, name='bn_3x3', scale=True)
                conv_1x1_input_tensor = tf.keras.layers.ZeroPadding2D(padding=0).call(inputs=input_tensor) if \
                    padding == 'VALID' else input_tensor
                conv_1x1 = self.conv2d(
                    inputdata=conv_1x1_input_tensor,
                    out_channel=output_channels,
                    kernel_size=1,
                    padding=padding,
                    stride=stride,
                    use_bias=False,
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

    def save_trained_model(self, weights_path, name, dump_to_json=False):
        """

        :param weights_path:
        :param name:
        :param dump_to_json:
        :return:
        """
        input_tensor = tf.placeholder(
            dtype=tf.float32,
            shape=[None, INPUT_TENSOR_SIZE[0], INPUT_TENSOR_SIZE[1], INPUT_CHANNELS],
            name='input_tensor'
        )

        _ = self.conv_block(
            input_tensor=input_tensor,
            output_channels=OUTPUT_CHANNELS,
            name=name,
            apply_reparam=False,
            stride=STRIDE,
            padding=PADDING
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
            corresponding_params = []
            for trainned_param_name, trainned_param_value in repvgg_trainned_params.items():
                corresponding_params.append((trainned_param_name, trainned_param_value))

            output_kernel, output_bias = self._fuse_conv_block_params(corresponding_params)

            if 'bias' in param_name or 'b' in param_name.split('/')[-1]:
                converted_params[param_name] = output_bias
            elif 'kernel' in param_name or 'W' in param_name.split('/')[-1]:
                converted_params[param_name] = output_kernel
            else:
                raise ValueError('Unrecognized param: {:s}'.format(param_name))

        return converted_params

    def compute_loss(self, input_tensor, label, name, reuse=False):
        """

        :param input_tensor:
        :param label:
        :param name:
        :param reuse:
        :return:
        """
        pass

    def inference(self, input_tensor, name, reuse=False):
        """

        :param input_tensor:
        :param name:
        :param reuse:
        :return:
        """
        pass


def get_model(phase, cfg):
    """

    :param phase:
    :param cfg:
    :return:
    """
    return RepVggTest(phase=phase, cfg=cfg)


def _init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_ckpt', type=str, help='The trainned repvgg model ckpt weights path')
    parser.add_argument('--deploy_ckpt', type=str)

    return parser.parse_args()


def save_train_params():
    """

    :return:
    """
    tf.reset_default_graph()
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_repvgg.yaml')
    model = get_model(phase=PHASE, cfg=cfg)
    input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[None, INPUT_TENSOR_SIZE[0], INPUT_TENSOR_SIZE[1], INPUT_CHANNELS],
        name='input_tensor'
    )

    _ = model.conv_block(
        input_tensor=input_tensor,
        output_channels=OUTPUT_CHANNELS,
        name="RepVgg",
        apply_reparam=False,
        stride=STRIDE,
        padding=PADDING
    )

    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(cfg.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess=sess, save_path='./log/train/repvgg_train.ckpt')


def convert_repvgg_params():
    """

    :return:
    """
    tf.reset_default_graph()
    args = _init_args()

    # init compute graph
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_repvgg.yaml')
    model = get_model(phase=PHASE, cfg=cfg)

    # save trainned repvgg params into json
    repvgg_trainned_params = dict()
    trained_params = model.save_trained_model(weights_path=args.train_ckpt, name='RepVgg')
    for param_name, param_value in trained_params.items():
        repvgg_trainned_params[param_name] = np.array(param_value, dtype=np.float32)

    # build repvgg inference compute graph
    tf.reset_default_graph()
    test_input_tensor = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, INPUT_CHANNELS], name='test_input')
    _ = model.conv_block(
        input_tensor=test_input_tensor,
        output_channels=OUTPUT_CHANNELS,
        name='RepVgg',
        apply_reparam=True,
        stride=STRIDE,
        padding=PADDING
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

        saver.save(sess, save_path=args.deploy_ckpt)

    print('Complete')


def check_converted_model():
    """

    :return:
    """
    tf.reset_default_graph()
    args = _init_args()

    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_repvgg.yaml')
    model = get_model(phase=PHASE, cfg=cfg)
    test_input_ndaray = np.random.random(
        (1, INPUT_TENSOR_SIZE[0], INPUT_TENSOR_SIZE[1], INPUT_CHANNELS)
    ).astype(np.float32)

    # init trained compute graph
    test_input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[1, INPUT_TENSOR_SIZE[0], INPUT_TENSOR_SIZE[1], INPUT_CHANNELS],
        name='test_input'
    )
    logits = model.conv_block(
        input_tensor=test_input_tensor,
        output_channels=OUTPUT_CHANNELS,
        name="RepVgg",
        apply_reparam=False,
        stride=STRIDE,
        padding=PADDING
    )

    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(cfg.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=args.train_ckpt)

        train_logits = sess.run(logits, feed_dict={test_input_tensor: test_input_ndaray})
        print(train_logits)
        print('*' * 100)

    # reset compute graph
    tf.reset_default_graph()
    test_input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[1, INPUT_TENSOR_SIZE[0], INPUT_TENSOR_SIZE[1], INPUT_CHANNELS],
        name='test_input'
    )
    logits = model.conv_block(
        input_tensor=test_input_tensor,
        output_channels=OUTPUT_CHANNELS,
        name='RepVgg',
        apply_reparam=True,
        stride=STRIDE,
        padding=PADDING
    )
    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        saver.restore(sess=sess, save_path=args.deploy_ckpt)

        deploy_logits = sess.run(logits, feed_dict={test_input_tensor: test_input_ndaray})
        print(deploy_logits)
        print('*' * 100)

    print('========================== The diff is')
    print(((train_logits - deploy_logits) ** 2).sum())


if __name__ == '__main__':
    """
    
    """
    save_train_params()

    convert_repvgg_params()

    check_converted_model()
