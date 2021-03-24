#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/24 下午2:07
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : base_dataset_provider.py
# @IDE: PyCharm
"""
base dataset reader
"""
from abc import ABCMeta
from abc import abstractmethod
import os.path as ops

import cv2
import numpy as np
import tqdm
import loguru

from local_utils.augment_utils import augmentation_utils as aug

LOG = loguru.logger


class DataSet(metaclass=ABCMeta):
    """

    """
    def __init__(self, image_label_infos, cfg):
        """

        :param image_label_infos: [(image_path1, class_id1), (image_path2, class_id2)]
        :param cfg
        """
        self._cfg = cfg
        self._image_label_infos = image_label_infos
        self._epoch_nums = self._cfg.TRAIN.EPOCH_NUMS
        self._batch_size = self._cfg.TRAIN.BATCH_SIZE
        self._batch_count = 0
        self._sample_nums = len(image_label_infos)
        self._num_batchs = int(np.ceil(self._sample_nums / self._batch_size))
        self._use_one_hot = self._cfg.DATASET.USE_ONE_HOT_LABEL

    def _load_batch_images(self, image_label_info):
        """

        :param image_label_info:
        :return:
        """
        src_images = []
        labels = []

        for info in image_label_info:
            src_image = cv2.imread(info[0], cv2.IMREAD_COLOR)
            if src_image is None:
                return None, None
            src_images.append(src_image)
            if not self._use_one_hot:
                labels.append(int(info[1]))
            else:
                one_hot_label = [0.0] * self._cfg.DATASET.NUM_CLASSES
                one_hot_label[int(info[1])] = 1.0
                labels.append(one_hot_label)

        return src_images, labels

    def _multiprocess_preprocess_images(self, src_images):
        """

        :param src_images:
        :return:
        """
        output_src_images = []

        for index, src_image in enumerate(src_images):
            output_src_image = aug.preprocess_image(
                src_image,
                cfg=self._cfg
            )
            output_src_images.append(output_src_image)
        return output_src_images

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

    def __next__(self):
        """

        :return:
        """
        if self._batch_count < self._num_batchs:
            batch_image_paths = self._image_label_infos[self._batch_count:self._batch_count + self._batch_size]
            batch_src_images, batch_labels = self._load_batch_images(batch_image_paths)
            if batch_src_images is None or batch_labels is None:
                return None, None
            batch_src_images = self._multiprocess_preprocess_images(
                batch_src_images
            )
            self._batch_count += 1

            return batch_src_images, batch_labels
        else:
            self._batch_count = 0
            np.random.shuffle(self._image_label_infos)
            raise StopIteration


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
            )
            self._val_dataset = DataSet(
                image_label_infos=self._val_label_image_infos,
                cfg=cfg,
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
