#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 下午3:09
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : ilsvrc_2012_tf_provider.py
# @IDE: PyCharm
"""
ilsvrc dataset reader
"""
import time
import os.path as ops

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from data_provider import base_dataset_tf_provider
from local_utils import config_utils


class IlsvrcDatasetTfProvider(base_dataset_tf_provider.DataSetProvider):
    """
    Ilsvrc dataset reader
    """
    def __init__(self, cfg):
        """

        """
        super(IlsvrcDatasetTfProvider, self).__init__(cfg=cfg)

    def _load_train_val_image_index(self):
        """

        :return:
        """

        with open(self._train_image_index_file_path, 'r') as file:
            for line in file:
                info = line.rstrip('\r').rstrip('\n').strip(' ').split()
                train_src_image_path = info[0]
                label_id = info[1]
                assert ops.exists(train_src_image_path), '{:s} not exist'.format(train_src_image_path)

                self._train_label_image_infos.append([train_src_image_path, label_id])

        with open(self._val_image_index_file_path, 'r') as file:
            for line in file:
                info = line.rstrip('\r').rstrip('\n').strip(' ').split()
                val_src_image_path = info[0]
                val_label_id = info[1]
                assert ops.exists(val_src_image_path), '{:s} not exist'.format(val_src_image_path)

                self._val_label_image_infos.append([val_src_image_path, val_label_id])
        return


def get_provider(cfg):
    """

    :return:
    """
    return IlsvrcDatasetTfProvider(cfg=cfg)


def _test():
    """

    :return:
    """
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_xception.yaml')
    reader = IlsvrcDatasetTfProvider(cfg=cfg)
    if not reader.successfully_initialized:
        print('Dataset reader not successfully initialized')
        return
    train_dataset = reader.train_dataset
    val_dataset = reader.val_dataset

    samples = val_dataset.next_batch()
    src_images = samples['images']
    src_labels = samples['labels']
    aug_images = samples['aug_images']
    src_images_path = samples['images_path']

    count = 1
    with tf.Session() as sess:
        while True:
            try:
                t_start = time.time()
                aug_images_value, labels_value, image_paths_value, src_images_value = sess.run(
                    [aug_images, src_labels, src_images_path, src_images]
                )
                print('Iter: {:d}, cost time: {:.5f}s'.format(count, time.time() - t_start))
                count += 1
                aug_image_recover = np.array((aug_images_value[0] + 1.0) * 127.5, dtype=np.uint8)
                src_image = np.array(src_images_value[0], dtype=np.uint8)

                # print(image_paths_value[0].decode('utf-8'))
                # print(labels_value[0])
                # plt.figure('src')
                # plt.imshow(src_image)
                # plt.figure('aug_recover')
                # plt.imshow(aug_image_recover)
                #
                # plt.show()
                # raise ValueError
            except tf.errors.OutOfRangeError as err:
                print(err)


if __name__ == '__main__':
    """
    test code
    """
    _test()
