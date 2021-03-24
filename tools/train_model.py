#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/24 下午4:03
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : train_model.py
# @IDE: PyCharm
"""

"""
import os.path as ops
import argparse

import trainner
from local_utils import config_utils
from local_utils import log_util


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='Net architecture used for image classification', default='xception')
    parser.add_argument('--dataset', type=str, help='Dataset name', default='ilsvrc_2012')

    return parser.parse_args()


def train():
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
    log_util.get_logger(cfg=cfg)
    net_trainner = trainner.get_trainner(cfg=cfg)
    net_trainner.train()

    return


if __name__ == '__main__':
    """
    main func
    """
    train()
