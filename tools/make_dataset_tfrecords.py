#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/4/14 上午10:59
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : make_dataset_tfrecords.py
# @IDE: PyCharm
"""
Make dataset tfrecords
"""
import argparse
import os
import os.path as ops
import time
import collections
import random

import tqdm
import tensorflow as tf
import loguru
import six
import cv2

from local_utils import config_utils
from local_utils.augment_utils import augmentation_utils as aug

LOG = loguru.logger
LOG.add(
    './log/make_dataset_tfrecords.log',
    level='INFO',
)


def _int64_list_feature(values):
    """

    :param values:
    :return:
    """
    if not isinstance(values, collections.Iterable):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
    """

    :param values:
    :return:
    """
    def _norm2bytes(value):
        return value.encode() if isinstance(value, str) and six.PY3 else value

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[_norm2bytes(values)]))


def _init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='xception')
    parser.add_argument('--dataset', type=str, default='ilsvrc_2012')

    return parser.parse_args()


def _generate_single_example(src_image_path, src_image_label, cfg):
    """

    :param src_image_path:
    :param src_image_label:
    :param cfg:
    :return:
    """
    source_image = cv2.imread(src_image_path, cv2.IMREAD_COLOR)
    source_image = aug.resize(img=source_image, cfg=cfg)
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    source_image_raw = source_image.tobytes()

    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image_raw': _bytes_list_feature(source_image_raw),
                'image_label': _int64_list_feature(src_image_label),
                'image_path': _bytes_list_feature(src_image_path)
            }
        )
    )

    return example.SerializeToString()


def _parse_image_index_file(image_index_file_path):
    """

    :param image_index_file_path:
    :return:
    """
    assert ops.exists(image_index_file_path), '{:s} not exist'.format(image_index_file_path)

    parse_result = []
    with open(image_index_file_path, 'r') as file:
        for line in file:
            info = line.rstrip('\r').rstrip('\n').strip(' ').split()
            assert len(info) == 2
            parse_result.append(info)
    random.shuffle(parse_result)
    return parse_result


def _make_dataset_tf_records(cfg, save_dir, dataset_flag='train', split_ratio=None):
    """

    :param cfg:
    :param save_dir:
    :param dataset_flag
    :param split_ratio:
    :return:
    """
    if dataset_flag == 'train':
        image_index_file_path = cfg.DATASET.TRAIN_FILE_LIST
    elif dataset_flag == 'test':
        image_index_file_path = cfg.DATASET.TEST_FILE_LIST
    elif dataset_flag == 'val':
        image_index_file_path = cfg.DATASET.VAL_FILE_LIST
    else:
        raise ValueError

    image_index_info = _parse_image_index_file(image_index_file_path)
    dataset_name = cfg.DATASET.DATASET_NAME

    if split_ratio is not None:
        image_index_info_split = []
        for index in range(0, len(image_index_info), split_ratio):
            image_index_info_split.append(image_index_info[index:index + split_ratio])

        tqbar = tqdm.tqdm(total=len(image_index_info))
        for index, example_infos in enumerate(image_index_info_split):
            tfrecords_output_name = '{:s}_{:s}_{:d}_{:d}.tfrecords'.format(
                dataset_name, dataset_flag, index, len(example_infos)
            )
            tfrecords_output_path = ops.join(save_dir, tfrecords_output_name)

            with tf.io.TFRecordWriter(tfrecords_output_path) as writer:
                for example_index, example_info in enumerate(example_infos):
                    t_start = time.time()
                    src_image_path = example_info[0]
                    src_image_label = int(example_info[1])
                    example = _generate_single_example(
                        src_image_path=src_image_path,
                        src_image_label=src_image_label,
                        cfg=cfg
                    )
                    writer.write(example)
                    t_cost = time.time() - t_start
                    tqbar.update(1)
                    tqbar.set_description('write single example cost time: {:.5f}s'.format(t_cost))
        tqbar.close()
    else:
        tfrecords_output_name = '{:s}_{:s}.tfrecords'.format(dataset_name, dataset_flag)
        tfrecords_output_path = ops.join(save_dir, tfrecords_output_name)

        tqbar = tqdm.tqdm(image_index_info)
        with tf.python_io.TFRecordWriter(tfrecords_output_path) as writer:
            for example_info in tqbar:
                t_start = time.time()
                src_image_path = example_info[0]
                src_image_label = example_info[1]
                example = _generate_single_example(
                    src_image_path=src_image_path,
                    src_image_label=src_image_label,
                    cfg=cfg
                )
                writer.write(example)
                t_cost = time.time() - t_start
                tqbar.set_description('write single example cost time: {:.5f}s'.format(t_cost))
        tqbar.close()

    LOG.info('Generate {:s} tfrecords for {:s} complete'.format(dataset_flag, dataset_name))
    return


def main():
    """

    :return:
    """
    args = _init_args()

    net_name = args.net
    dataset_name = args.dataset
    config_file_name = '{:s}_{:s}.yaml'.format(dataset_name, net_name)
    config_file_path = ops.join('./config', config_file_name)
    if not ops.exists(config_file_path):
        raise ValueError('Config file path: {:s} not exist'.format(config_file_path))

    cfg = config_utils.get_config(config_file_path=config_file_path)

    tfrecords_save_dir = ops.join(cfg.DATASET.DATA_DIR, 'tfrecords')
    os.makedirs(tfrecords_save_dir, exist_ok=True)

    LOG.info('Start generate val dataset tfrecords for {:s}'.format(dataset_name))
    _make_dataset_tf_records(cfg=cfg, save_dir=tfrecords_save_dir, dataset_flag='val', split_ratio=10000)

    LOG.info('Start generate train dataset tfrecords for {:s}'.format(dataset_name))
    _make_dataset_tf_records(cfg=cfg, save_dir=tfrecords_save_dir, dataset_flag='train', split_ratio=50000)

    return


if __name__ == '__main__':
    """
    
    """
    main()
