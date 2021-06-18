#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 下午2:07
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : base_dataset_tf_provider_v1.py
# @IDE: PyCharm
"""
base dataset tensorflow reader
"""
from abc import ABCMeta
from abc import abstractmethod
import os.path as ops

import numpy as np
import loguru
import tensorflow as tf

from local_utils.augment_utils import augmentation_tf_utils as aug

LOG = loguru.logger


class DataSet(metaclass=ABCMeta):
    """

    """
    def __init__(self, tfrecord_paths, cfg, dataset_flag):
        """

        :param tfrecord_paths:
        :param cfg
        """
        self._cfg = cfg
        self._dataset_flag = dataset_flag
        self._epoch_nums = self._cfg.TRAIN.EPOCH_NUMS
        self._batch_size = self._cfg.TRAIN.BATCH_SIZE
        self._batch_count = 0
        self._tfrecord_file_paths = tfrecord_paths
        self._sample_nums = self._count_sample_nums()
        self._num_batchs = int(np.ceil(self._sample_nums / self._batch_size))
        self._use_one_hot = self._cfg.DATASET.USE_ONE_HOT_LABEL
        self._cpu_multi_process_nums = self._cfg.TRAIN.FAST_DATA_PROVIDER.MULTI_PROCESSOR_NUMS
        self._shuffle_buffer_size = self._cfg.TRAIN.FAST_DATA_PROVIDER.SHUFFLE_BUFFER_SIZE
        self._prefetch_size = self._cfg.TRAIN.FAST_DATA_PROVIDER.PREFETCH_SIZE

    def __len__(self):
        """

        :return:
        """
        return self._num_batchs

    def __iter__(self):
        """

        :return:
        """
        return self

    def _count_sample_nums(self):
        """

        :return:
        """
        if self._dataset_flag == 'train':
            sample_counts = len(open(self._cfg.DATASET.TRAIN_FILE_LIST).readlines())
        elif self._dataset_flag == 'val':
            sample_counts = len(open(self._cfg.DATASET.VAL_FILE_LIST).readlines())
        elif self._dataset_flag == 'test':
            sample_counts = len(open(self._cfg.DATASET.TEST_FILE_LIST).readlines())
        else:
            raise ValueError('Unsupported dataset flag: {:s}'.format(self._dataset_flag))

        return sample_counts

    def _decode_tf_sample(self, example):
        """

        :param example:
        :return:
        """
        features = tf.parse_single_example(
            example,
            # Defaults are not specified since both keys are required.
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'image_path': tf.FixedLenFeature([], tf.string),
                'image_label': tf.FixedLenFeature([], tf.int64)
            }
        )

        # decode gt image
        gt_image = tf.io.decode_raw(features['image_raw'], tf.uint8)
        gt_image = tf.reshape(
            gt_image,
            shape=[self._cfg.AUG.FIX_RESIZE_SIZE[1], self._cfg.AUG.FIX_RESIZE_SIZE[0], 3]
        )
        gt_image = tf.cast(gt_image, dtype=tf.float32)

        gt_label = tf.cast(features['image_label'], tf.int32)

        return gt_image, gt_label, features['image_path']

    def _augment_samples(self, source_image):
        """

        :param source_image:
        :return:
        """
        return aug.preprocess_image(source_image, cfg=self._cfg)

    def next_batch(self):
        """

        :return:
        """
        dataset = tf.data.TFRecordDataset(self._tfrecord_file_paths)
        dataset = dataset.map(
            map_func=self._decode_tf_sample,
            num_parallel_calls=self._cpu_multi_process_nums
        )
        dataset = dataset.map(
            lambda image, labels, images_path: {'images': image, 'images_path': images_path, 'labels': labels},
            num_parallel_calls=self._cpu_multi_process_nums
        )

        dataset = dataset.map(
            lambda d:
            (self._augment_samples(d['images']),
             d['images'],
             d['images_path'],
             d['labels']),
            num_parallel_calls=self._cpu_multi_process_nums
        )
        dataset = dataset.map(
            lambda aug_result, images, images_path, labels:
            {'aug_images': aug_result, 'images': images, 'images_path': images_path, 'labels': labels},
            num_parallel_calls=self._cpu_multi_process_nums
        )
        dataset = dataset.shuffle(self._shuffle_buffer_size)

        dataset = dataset.padded_batch(
            batch_size=self._batch_size,
            padded_shapes={
                'aug_images': [None, None, 3],
                'images': [None, None, 3],
                'images_path': [],
                'labels': [],
            },
            padding_values={
                'aug_images': 0.0,
                'images': 0.0,
                'images_path': 'None',
                'labels': -1,
            },
            drop_remainder=True
        )
        dataset = dataset.repeat()
        dataset = dataset.prefetch(self._prefetch_size)

        iterator = dataset.make_one_shot_iterator()

        return iterator.get_next(name='{:s}_iterator_getnext'.format(self._dataset_flag))


class DataSetProvider(metaclass=ABCMeta):
    """

    """
    def __init__(self, cfg):
        """

        :param cfg:
        """
        self._cfg = cfg
        self._dataset_dir = self._cfg.DATASET.DATA_DIR
        self._batch_size = self._cfg.TRAIN.BATCH_SIZE
        self._use_one_hot_labels = self._cfg.DATASET.USE_ONE_HOT_LABEL
        self._successfully_init = False

        assert ops.exists(self._dataset_dir), 'Dataset dir: {:s} not exist'.format(self._dataset_dir)

        self._train_label_image_infos = []
        self._val_label_image_infos = []
        self._load_train_val_image_index()
        np.random.shuffle(self._train_label_image_infos)
        np.random.shuffle(self._val_label_image_infos)

        self._train_dataset = DataSet(
            tfrecord_paths=self._train_label_image_infos,
            cfg=cfg,
            dataset_flag='train'
        )
        self._val_dataset = DataSet(
            tfrecord_paths=self._val_label_image_infos,
            cfg=cfg,
            dataset_flag='val'
        )
        self._successfully_init = True

    @abstractmethod
    def _load_train_val_image_index(self):
        """
        parse train/val image index file to fill in train/val label image infos
        :return:
        """
        pass

    @property
    def train_dataset(self):
        """

        :param
        :return:
        """
        return self._train_dataset

    @property
    def val_dataset(self):
        """

        :return:
        """
        return self._val_dataset

    @property
    def successfully_initialized(self):
        """

        :return:
        """
        return self._successfully_init
