#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/6/28 下午2:21
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : convert_model.py
# @IDE: PyCharm
"""
Convert model weights
"""
import os.path as ops
import argparse

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow import saved_model as sm
import tf2onnx

import cls_model_zoo
from local_utils import config_utils


def _build_model_graph_session(model_cfg):
    """

    :param model_cfg:
    :return:
    """
    cfg = model_cfg
    net = cls_model_zoo.get_model(cfg=cfg, phase='test')

    # construct compute graph
    input_tensor_width = cfg.AUG.EVAL_CROP_SIZE[0]
    input_tensor_height = cfg.AUG.EVAL_CROP_SIZE[1]
    input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[1, input_tensor_height, input_tensor_width, 3],
        name='input_tensor'
    )
    logits = net.inference(input_tensor=input_tensor, name=cfg.MODEL.MODEL_NAME, reuse=False)
    _ = tf.nn.softmax(logits, name='output_tensor')

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.85
    sess_config.gpu_options.allow_growth = False
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)
    gd = graph_util.remove_training_nodes(sess.graph_def)

    return sess, gd


def _convert_ckpt_to_frozen_pb(net_name, dataset_name, ckpt_file_path, frozen_pb_save_path):
    """

    :param net_name:
    :param dataset_name:
    :param ckpt_file_path:
    :param frozen_pb_save_path:
    :return:
    """
    config_file_name = '{:s}_{:s}.yaml'.format(dataset_name, net_name)
    config_file_path = ops.join('./config', config_file_name)
    if not ops.exists(config_file_path):
        raise ValueError('Config file path: {:s} not exist'.format(config_file_path))
    cfg = config_utils.get_config(config_file_path=config_file_path)
    sess, gd = _build_model_graph_session(model_cfg=cfg)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            cfg.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
    # define saver
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess, ckpt_file_path)  # variables

        # generate protobuf
        converted_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def=gd,
            output_node_names=["output_tensor"]
        )
        tf.train.write_graph(converted_graph_def, './', frozen_pb_save_path, as_text=False)

    print('Convert ckpt weights to frozen pb completed!!!')
    return


def _convert_ckpt_to_tf_saved_model(net_name, dataset_name, ckpt_fle_path, saved_model_export_dir):
    """

    :param net_name:
    :param dataset_name:
    :param ckpt_fle_path:
    :param saved_model_export_dir:
    :return:
    """
    config_file_name = '{:s}_{:s}.yaml'.format(dataset_name, net_name)
    config_file_path = ops.join('./config', config_file_name)
    if not ops.exists(config_file_path):
        raise ValueError('Config file path: {:s} not exist'.format(config_file_path))
    cfg = config_utils.get_config(config_file_path=config_file_path)
    sess, gd = _build_model_graph_session(model_cfg=cfg)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            cfg.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
    # define saver
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():

        saver.restore(sess=sess, save_path=ckpt_fle_path)

        # set model save builder
        saved_builder = sm.Builder(saved_model_export_dir)

        # add tensor need to be saved
        saved_input_tensor = sm.build_tensor_info(sess.graph.get_tensor_by_name("input_tensor"))
        saved_output_tensor = sm.build_tensor_info(sess.graph.get_tensor_by_name("output_tensor"))

        # build SignatureDef protobuf
        signatur_def = sm.build_signature_def(
            inputs={'input_tensor': saved_input_tensor},
            outputs={'output_tensor': saved_output_tensor},
            method_name=tf.saved_model.CLASSIFY_METHOD_NAME
        )

        # add graph into MetaGraphDef protobuf
        saved_builder.add_meta_graph_and_variables(
            sess,
            tags=[sm.SERVING],
            signature_def_map={'classify_result': signatur_def}
        )

        # save model
        saved_builder.save()

    print('Convert ckpt weights to tf saved model completed!!!')
    return


def _convert_ckpt_to_onnx(net_name, dataset_name, ckpt_file_path, onnx_save_path):
    """

    :param net_name:
    :param dataset_name:
    :param ckpt_file_path:
    :param onnx_save_path:
    :return:
    """
    config_file_name = '{:s}_{:s}.yaml'.format(dataset_name, net_name)
    config_file_path = ops.join('./config', config_file_name)
    if not ops.exists(config_file_path):
        raise ValueError('Config file path: {:s} not exist'.format(config_file_path))
    cfg = config_utils.get_config(config_file_path=config_file_path)
    sess, gd = _build_model_graph_session(model_cfg=cfg)

    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            cfg.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    with sess.as_default():
        saver.restore(sess=sess, save_path=ckpt_file_path)
        onnx_graph = tf2onnx.tfonnx.process_tf_graph(
            gd,
            input_names=["input_tensor:0"],
            output_names=["output_tensor:0"]
        )
        model_proto = onnx_graph.make_model('{:s}_{:s}_onnx_model'.format(dataset_name, net_name))
        with open(onnx_save_path, "wb") as f:
            f.write(model_proto.SerializeToString())

    print('Convert ckpt weights to onnx completed!!!')
    return


if __name__ == '__main__':
    """
    
    """
    # _convert_ckpt_to_onnx(
    #     net_name='mobilenetv2',
    #     dataset_name='ilsvrc_2012',
    #     ckpt_file_path='./model_weights/tf_checkpoint/mobilenetv2.ckpt-128',
    #     onnx_save_path='./model_weights/onnx/mobilenetv2.onnx'
    # )

    _convert_ckpt_to_frozen_pb(
        net_name='mobilenetv2',
        dataset_name='ilsvrc_2012',
        ckpt_file_path='./model_weights/tf_checkpoint/mobilenetv2.ckpt-128',
        frozen_pb_save_path='./model_weights/onnx/mobilenetv2.pb'
    )
