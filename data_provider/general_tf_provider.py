#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 下午3:09
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : ilsvrc_2012_tf_provider_v1.py
# @IDE: PyCharm
"""
general tf dataset reader
"""
import time
import os.path as ops
import glob

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
        tfrecords_dir = ops.join(self._dataset_dir, 'tfrecords')
        if not ops.exists(tfrecords_dir):
            return

        self._train_label_image_infos = glob.glob('{:s}/{:s}_train*.tfrecords'.format(
            tfrecords_dir, self._cfg.DATASET.DATASET_NAME)
        )
        self._val_label_image_infos = glob.glob('{:s}/{:s}_val*.tfrecords'.format(
            tfrecords_dir, self._cfg.DATASET.DATASET_NAME)
        )

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
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_resnet.yaml')
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

                print(image_paths_value[0].decode('utf-8'))
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
