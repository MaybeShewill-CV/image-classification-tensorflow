#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/6/17 下午1:48
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : prepare_cls_task.py
# @IDE: PyCharm
"""
prepare classification task
"""
import argparse
import os
import os.path as ops
import codecs
import yaml
import collections
import json
import glob
from ast import literal_eval

import loguru
import tqdm

LOG = loguru.logger
LOG.add(
    './log/prepare_cls_task.log',
    level='INFO',
)


def _check_src_image_dir_complete(task_dir):
    """

    :param task_dir:
    :return:
    """
    train_src_image_dir = ops.join(task_dir, 'src_image', 'train')
    if not ops.exists(train_src_image_dir):
        LOG.error('Source training image dir: {:s} not exist'.format(train_src_image_dir))
        return False

    test_src_image_dir = ops.join(task_dir, 'src_image', 'test')
    if not ops.exists(train_src_image_dir):
        LOG.error('Source testing image dir: {:s} not exist'.format(test_src_image_dir))
        return False

    val_src_image_dir = ops.join(task_dir, 'src_image', 'val')
    if not ops.exists(train_src_image_dir):
        LOG.error('Source val image dir: {:s} not exist'.format(val_src_image_dir))
        return False

    return True


def _check_image_classes_consistency(task_dir):
    """

    :param task_dir:
    :return:
    """
    if not _check_src_image_dir_complete(task_dir=task_dir):
        return False

    train_src_image_dir = ops.join(task_dir, 'src_image', 'train')
    train_cls_set = set(os.listdir(train_src_image_dir))

    test_src_image_dir = ops.join(task_dir, 'src_image', 'test')
    test_cls_set = set(os.listdir(test_src_image_dir))

    val_src_image_dir = ops.join(task_dir, 'src_image', 'val')
    val_cls_set = set(os.listdir(val_src_image_dir))

    if train_cls_set == test_cls_set == val_cls_set:
        return True
    else:
        LOG.error('Check image classes consistency failed. Train, '
                  'test, val dataset should share the same image classes, please check.')
        return False


def _generate_class_id_map_file(task_dir):
    """

    :param task_dir:
    :return:
    """
    if not _check_image_classes_consistency(task_dir=task_dir):
        LOG.error('Generate class id map file failed')
        return False

    train_src_image_dir = ops.join(task_dir, 'src_image', 'train')
    cls_list = os.listdir(train_src_image_dir)
    cls_map = collections.OrderedDict()
    for cls_id, cls_name in enumerate(cls_list):
        cls_map[cls_name] = cls_id

    with open(ops.join(task_dir, 'class_id_map.json'), 'w') as file:
        json.dump(cls_map, file)

    return cls_map


def _make_image_index_file(task_dir):
    """

    :param task_dir:
    :return:
    """
    if not ops.exists(task_dir):
        LOG.error('Task dir: {:s} not exist'.format(task_dir))
        return False

    image_index_file_dir = ops.join(task_dir, 'image_index_file')
    os.makedirs(image_index_file_dir, exist_ok=True)

    cls_id_map_file = ops.join(task_dir, 'class_id_map.json')
    with open(cls_id_map_file, 'r') as file:
        cls_id_map = json.load(file)

    # generate train index file
    train_src_image_dir = ops.join(task_dir, 'src_image', 'train')
    train_src_image_paths = glob.glob('{:s}/*'.format(train_src_image_dir))
    train_src_image_paths = filter(
        lambda x: ops.split(x)[1].endswith(('jpg', 'png', 'jpeg', 'JPEG', 'tif', 'tiff')),
        train_src_image_paths
    )
    train_image_file_index_info = []
    train_image_paths_tqdm = tqdm.tqdm(train_src_image_paths)
    train_image_paths_tqdm.set_description('making train image index file')
    for img_path in train_image_paths_tqdm:
        folder_name = ops.split(ops.split(img_path)[0])[1]
        if folder_name not in cls_id_map:
            LOG.error('Class name: {:s} not existed in class id map file')
            return False
        cls_id = cls_id_map[folder_name]
        train_image_file_index_info.append('{:s} {:d}'.format(img_path, cls_id))

    # generate test index file
    test_src_image_dir = ops.join(task_dir, 'src_image', 'test')
    test_src_image_paths = glob.glob('{:s}/*'.format(test_src_image_dir))
    test_src_image_paths = filter(
        lambda x: ops.split(x)[1].endswith(('jpg', 'png', 'jpeg', 'JPEG', 'tif', 'tiff')),
        test_src_image_paths
    )
    test_image_file_index_info = []
    test_image_paths_tqdm = tqdm.tqdm(test_src_image_paths)
    test_image_paths_tqdm.set_description('making test image index file')
    for img_path in test_image_paths_tqdm:
        folder_name = ops.split(ops.split(img_path)[0])[1]
        if folder_name not in cls_id_map:
            LOG.error('Class name: {:s} not existed in class id map file')
            return False
        cls_id = cls_id_map[folder_name]
        test_image_file_index_info.append('{:s} {:d}'.format(img_path, cls_id))

    # generate val index file
    val_src_image_dir = ops.join(task_dir, 'src_image', 'val')
    val_src_image_paths = glob.glob('{:s}/*'.format(val_src_image_dir))
    val_src_image_paths = filter(
        lambda x: ops.split(x)[1].endswith(('jpg', 'png', 'jpeg', 'JPEG', 'tif', 'tiff')),
        val_src_image_paths
    )
    val_image_file_index_info = []
    val_image_paths_tqdm = tqdm.tqdm(val_src_image_paths)
    val_image_paths_tqdm.set_description('making val image index file')
    for img_path in val_image_paths_tqdm:
        folder_name = ops.split(ops.split(img_path)[0])[1]
        if folder_name not in cls_id_map:
            LOG.error('Class name: {:s} not existed in class id map file')
            return False
        cls_id = cls_id_map[folder_name]
        val_image_file_index_info.append('{:s} {:d}'.format(img_path, cls_id))

    with open(ops.join(image_index_file_dir, 'train.txt'), 'w') as file:
        file.write('\n'.join(train_image_file_index_info))
    with open(ops.join(image_index_file_dir, 'test.txt'), 'w') as file:
        file.write('\n'.join(test_image_file_index_info))
    with open(ops.join(image_index_file_dir, 'val.txt'), 'w') as file:
        file.write('\n'.join(val_image_file_index_info))

    LOG.info('Generate image index file complete')
    return True


def _generate_template_model_cfg_file(task_dir, net_name, dataset_name):
    """

    :param task_dir:
    :param net_name:
    :param dataset_name:
    :return:
    """
    if net_name not in ['darknet53', 'densenet', 'inceptionv4', 'mobilenetv2', 'res2net',
                        'resnet', 'se-resnet', 'vgg', 'xception']:
        raise NotImplementedError('Not support model: {:s}'.format(net_name))

    darknet53_model_cfg = {
        'MODEL_NAME': 'darknet53'
    }
    densenet_model_cfg = {
        'MODEL_NAME': 'densenet',
        'DENSENET': {
            'GROWTH_RATE': 32,
            'BLOCK_NUMS': 4,
            'ENABLE_BC': True,
            'BC_THETA': 0.5,
            'COMPOSITE_CHANNELS': 128,
            'NET_SIZE': 121,
        }
    }
    inceptionv4_model_cfg = {
        'MODEL_NAME': 'inceptionv4',
        'INCEPTION': {
            'REDUCTION_A_PARAMS': {
                'K': 192,
                'L': 224,
                'M': 256,
                'N': 384,
            }
        }
    }
    mobilenetv2_model_cfg = {
        'MODEL_NAME': 'mobilenetv2'
    }
    res2net_model_cfg = {
        'MODEL_NAME': 'res2net',
        'RES2NET': {
            'NET_SIZE': 50,
            'BOTTLENECK_SCALE': 4,
            'BOTTLENECK_BASE_WIDTH': 26,
        }
    }
    resnet_model_cfg = {
        'MODEL_NAME': 'resnet',
        'RESNET': {
            'NET_SIZE': 50
        }
    }
    seresnet_model_cfg = {
        'MODEL_NAME': 'se-resnet',
        'SE_RESNET': {
            'NET_SIZE': 50,
            'REDUCTION': 16
        }
    }
    vgg_model_cfg = {
        'MODEL_NAME': 'vgg',
        'VGG': {
            'NET_SIZE': 16
        }
    }
    xception_model_cfg = {
        'MODEL_NAME': 'xception',
        'XCEPTION': {
            'DEPTH_MULTIPLIER': 1,
            'LAYERS': 39
        }
    }

    if net_name == 'darknet53':
        model_cfg = darknet53_model_cfg
    elif net_name == 'densenet':
        model_cfg = densenet_model_cfg
    elif net_name == 'inceptionv4':
        model_cfg = inceptionv4_model_cfg
    elif net_name == 'mobilenetv2':
        model_cfg = mobilenetv2_model_cfg
    elif net_name == 'res2net':
        model_cfg = res2net_model_cfg
    elif net_name == 'resnet':
        model_cfg = resnet_model_cfg
    elif net_name == 'se-resnet':
        model_cfg = seresnet_model_cfg
    elif net_name == 'vgg':
        model_cfg = vgg_model_cfg
    elif net_name == 'xception':
        model_cfg = xception_model_cfg
    else:
        raise NotImplementedError('Not support model: {:s}'.format(net_name))



