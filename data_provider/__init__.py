#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/18 下午7:12
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : __init__.py.py
# @IDE: PyCharm
"""
dataset provider
"""
import os.path as ops
import importlib


def get_dataset_provider(cfg):
    """

    :param cfg:
    :return:
    """
    dataset_name = cfg.DATASET.DATASET_NAME
    module_dir_name = ops.dirname(__file__)
    module_dir_name = ops.split(module_dir_name)[-1]
    if cfg.TRAIN.FAST_DATA_PROVIDER.ENABLE:
        module_name = '{:s}.{:s}_tf_provider'.format(module_dir_name, dataset_name)
    else:
        module_name = '{:s}.{:s}_provider'.format(module_dir_name, dataset_name)
    mod = importlib.import_module(module_name)

    return getattr(mod, "get_provider")(cfg=cfg)
