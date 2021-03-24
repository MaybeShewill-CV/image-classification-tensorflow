#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 下午3:09
# @Author  : LuoYao
# @Site    : ICode
# @File    : lutao_reader.py
# @IDE: PyCharm
"""
lutao dataset reader
"""
import os.path as ops

import tqdm

from data_provider import base_dataset_reader
from local_utils import config_utils


class IlsvrcDatasetProvider(base_dataset_reader.DataSetProvider):
    """
    Ilsvrc dataset reader
    """
    def __init__(self, cfg):
        """

        """
        super(IlsvrcDatasetProvider, self).__init__(cfg=cfg)

    def _load_train_val_image_index(self):
        """

        :return:
        """

        with open(self._train_image_index_file_path, 'r') as file:
            for line in file:
                info = line.rstrip('\r').rstrip('\n').strip(' ').split()
                train_src_image_path = ops.join(self._dataset_dir, 'ILSVRC2012_TRAIN', info[0])
                label_id = info[1]
                assert ops.exists(train_src_image_path), '{:s} not exist'.format(train_src_image_path)

                self._train_label_image_infos.append([train_src_image_path, label_id])

        with open(self._val_image_index_file_path, 'r') as file:
            for line in file:
                info = line.rstrip('\r').rstrip('\n').strip(' ').split()
                val_src_image_path = ops.join(self._dataset_dir, 'ILSVRC2012_IMG_VAL', info[0])
                val_label_id = info[1]
                assert ops.exists(val_src_image_path), '{:s} not exist'.format(val_src_image_path)

                self._val_label_image_infos.append([val_src_image_path, val_label_id])
        return


def get_provider(cfg):
    """

    :return:
    """
    return IlsvrcDatasetProvider(cfg=cfg)


def _test():
    """

    :return:
    """
    cfg = config_utils.get_config(config_file_path='./config/ilsvrc_2012_xception.yaml')
    reader = IlsvrcDatasetProvider(cfg=cfg)
    if not reader.successfully_initialized:
        print('Dataset reader not successfully initialized')
        return
    train_dataset = reader.train_dataset
    val_dataset = reader.val_dataset

    for train_samples in tqdm.tqdm(train_dataset):
        src_imgs = train_samples[0]
        src_labels = train_samples[1]

        if src_imgs is None or src_labels is None:
            print('Meet None')
            continue


if __name__ == '__main__':
    """
    test code
    """
    _test()
