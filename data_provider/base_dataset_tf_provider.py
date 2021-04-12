#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 下午2:07
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : base_dataset_tf_provider.py
# @IDE: PyCharm
"""
base dataset tensorflow reader
"""
from abc import ABCMeta
from abc import abstractmethod
import os.path as ops
import random

import numpy as np
import tqdm
import loguru
import tensorflow as tf

from local_utils.augment_utils import augmentation_tf_utils as aug

LOG = loguru.logger


class DataSet(metaclass=ABCMeta):
    """

    """
    def __init__(self, image_label_infos, cfg, dataset_flag):
        """

        :param image_label_infos: [(image_path1, class_id1), (image_path2, class_id2)]
        :param cfg
        """
        self._cfg = cfg
        self._dataset_flag = dataset_flag
        self._image_label_infos = image_label_infos
        self._epoch_nums = self._cfg.TRAIN.EPOCH_NUMS
        self._batch_size = self._cfg.TRAIN.BATCH_SIZE
        self._batch_count = 0
        self._sample_nums = len(image_label_infos)
        self._num_batchs = int(np.ceil(self._sample_nums / self._batch_size))
        self._use_one_hot = self._cfg.DATASET.USE_ONE_HOT_LABEL
        self._cpu_multi_process_nums = self._cfg.TRAIN.FAST_DATA_PROVIDER.MULTI_PROCESSOR_NUMS
        self._shuffle_buffer_size = self._cfg.TRAIN.FAST_DATA_PROVIDER.SHUFFLE_BUFFER_SIZE
        self._prefetch_size = self._cfg.TRAIN.FAST_DATA_PROVIDER.PREFETCH_SIZE
        self._sample_image_file_path = []
        self._sample_label = []
        random.shuffle(self._image_label_infos)
        for sample_info in self._image_label_infos:
            self._sample_image_file_path.append(sample_info[0])
            self._sample_label.append(sample_info[1])

    def _read_image_file(self, image_file_path):
        """

        :param image_file_path:
        :return:
        """
        image = tf.read_file(image_file_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, dtype=tf.float32)
        image = aug.resize(img=image, cfg=self._cfg)
        return image

    @classmethod
    def _read_label(cls, label):
        """

        :param label:
        :return:
        """
        cast_label = tf.py_func(lambda x: int(x), [label], tf.int64)
        return tf.cast(cast_label, tf.int32)

    def _augment_samples(self, source_image):
        """

        :param source_image:
        :return:
        """
        return aug.preprocess_image(source_image, cfg=self._cfg)

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

    def next_batch(self):
        """

        :return:
        """
        if self._sample_image_file_path is None or self._sample_label is None:
            raise ValueError('Sample index file paths was not successfully initialized, '
                             'please check your sample index file')

        auto_tune = tf.data.experimental.AUTOTUNE
        # Initialize dataset with file names
        dataset = tf.data.Dataset.from_tensor_slices(
            (self._sample_image_file_path, self._sample_label)
        )
        # Read image and point coordinates
        dataset = dataset.map(
            lambda images_path, labels: {'images_path': images_path, 'labels': labels},
            num_parallel_calls=auto_tune
        )
        dataset = dataset.map(
            lambda d: (self._read_image_file(d['images_path']), d['images_path'], self._read_label(d['labels'])),
            num_parallel_calls=auto_tune
        )
        dataset = dataset.map(
            lambda image, images_path, labels: (image, images_path, labels),
            num_parallel_calls=auto_tune
        )

        dataset = dataset.map(
            lambda image, images_path, labels: {'images': image, 'images_path': images_path, 'labels': labels},
            num_parallel_calls=auto_tune
        )
        dataset = dataset.map(
            lambda d:
            (self._augment_samples(d['images']),
             d['images'],
             d['images_path'],
             d['labels']),
            num_parallel_calls=auto_tune
        )
        dataset = dataset.map(
            lambda aug_result, images, images_path, labels:
            {'aug_images': aug_result, 'images': images, 'images_path': images_path, 'labels': labels},
            num_parallel_calls=auto_tune
        )
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

        # The shuffle transformation uses a finite-sized buffer to shuffle elements
        # in memory. The parameter is the number of elements in the buffer. For
        # completely uniform shuffling, set the parameter to be the same as the
        # number of elements in the dataset.
        dataset = dataset.shuffle(auto_tune)
        # repeat num epochs
        dataset = dataset.repeat(self._epoch_nums)

        dataset = dataset.prefetch(auto_tune)

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
        self._train_image_index_file_path = self._cfg.DATASET.TRAIN_FILE_LIST
        self._val_image_index_file_path = self._cfg.DATASET.VAL_FILE_LIST
        self._use_one_hot_labels = self._cfg.DATASET.USE_ONE_HOT_LABEL
        self._successfully_init = False

        assert ops.exists(self._dataset_dir), 'Dataset dir: {:s} not exist'.format(self._dataset_dir)
        assert ops.exists(self._train_image_index_file_path), 'Train image index file path: {:s} not exist'.format(
            self._train_image_index_file_path
        )
        assert ops.exists(self._val_image_index_file_path), 'Val image index file path: {:s} not exist'.format(
            self._val_image_index_file_path
        )

        self._train_label_image_infos = []
        self._val_label_image_infos = []
        self._load_train_val_image_index()
        np.random.shuffle(self._train_label_image_infos)
        np.random.shuffle(self._val_label_image_infos)

        if self._check_dataset_sample_info():
            self._train_dataset = DataSet(
                image_label_infos=self._train_label_image_infos,
                cfg=cfg,
                dataset_flag='train'
            )
            self._val_dataset = DataSet(
                image_label_infos=self._val_label_image_infos,
                cfg=cfg,
                dataset_flag='val'
            )
            self._successfully_init = True
        else:
            self._train_dataset = None
            self._val_dataset = None
            return

    def _check_dataset_sample_info(self):
        """

        :return:
        """
        if not self._train_label_image_infos:
            return False
        for train_sample_info in tqdm.tqdm(self._train_label_image_infos, desc='Check train dataset sample info'):
            if len(train_sample_info) < 2:
                LOG.error('Train sample info not enough, should be at least '
                          '[(image_path1, class_id1), (image_path2, class_id2), ...]')
                return False
            image_file_path = train_sample_info[0]
            image_class_id = train_sample_info[1]
            if not ops.exists(image_file_path) or not ops.isfile(image_file_path):
                LOG.error('Train sample: {:s} not exist'.format(image_file_path))
                return False
            if not image_class_id.isdigit():
                LOG.error('Class id shoule be digit')
                return False
            if int(image_class_id) >= self._cfg.DATASET.NUM_CLASSES:
                LOG.error('Class id should the smaller than class nums')
                return False

        if not self._val_label_image_infos:
            return False
        for val_sample_info in tqdm.tqdm(self._val_label_image_infos, desc='Check val dataset sample info'):
            if len(val_sample_info) < 2:
                LOG.error('Val sample info not enough, should be at least '
                          '[(image_path1, class_id1), (image_path2, class_id2), ...]')
                return False
            image_file_path = val_sample_info[0]
            image_class_id = val_sample_info[1]
            if not ops.exists(image_file_path) or not ops.isfile(image_file_path):
                LOG.error('Val sample: {:s} not exist'.format(image_file_path))
                return False
            if not image_class_id.isdigit():
                LOG.error('Class id shoule be digit')
                return False
            if int(image_class_id) >= self._cfg.DATASET.NUM_CLASSES:
                LOG.error('Class id should the smaller than class nums')
                return False

        return True

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
