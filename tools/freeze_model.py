#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/26 下午4:10
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : freeze_model.py
# @IDE: PyCharm
"""
freeze ckpt model file into pb model
"""
import os.path as ops
import argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util

import cls_model_zoo
from local_utils import config_utils


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='The network you used', default='xception')
    parser.add_argument('--dataset', type=str, help='The dataset', default='ilsvrc_2012')
    parser.add_argument('--weights_path', type=str, help='The ckpt weights file path')
    parser.add_argument('--pb_save_path', type=str, help='The converted pb file save path')

    return parser.parse_args()


def stats_graph(graph):
    """

    :param graph:
    :return:
    """
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

    return


def freeze_model():
    """

    :return:
    """
    args = init_args()

    net_name = args.net
    dataset_name = args.dataset
    config_file_name = '{:s}_{:s}.yaml'.format(dataset_name, net_name)
    config_file_path = ops.join('./config', config_file_name)
    if not ops.exists(config_file_path):
        raise ValueError('Config file path: {:s} not exist'.format(config_file_path))

    cfg = config_utils.get_config(config_file_path=config_file_path)
    net = cls_model_zoo.get_model(cfg=cfg, phase='test')

    # construct compute graph
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 224, 224, 3], name='input_tensor')
    logits = net.inference(input_tensor=input_tensor, name=cfg.MODEL.MODEL_NAME, reuse=False)
    prob_score = tf.nn.softmax(logits, name='output_tensor')

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            cfg.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.85
    sess_config.gpu_options.allow_growth = False
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    with sess.as_default():
        saver.restore(sess, args.weights_path)  # variables

        stats_graph(sess.graph)

        # generate protobuf
        converted_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def=sess.graph.as_graph_def(),
            output_node_names=["output_tensor"]
        )

        with tf.gfile.GFile(args.pb_save_path, "wb") as f:
            f.write(converted_graph_def.SerializeToString())

    print('Convert completed!!!')

    return


if __name__ == '__main__':
    """
    main func
    """
    freeze_model()
