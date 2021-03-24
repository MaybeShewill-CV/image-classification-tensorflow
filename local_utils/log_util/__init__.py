#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/14 下午8:18
# @Author  : LuoYao
# @Site    : ICode
# @File    : __init__.py.py
# @IDE: PyCharm
"""
config logger
"""
from local_utils.log_util import init_logger


def get_logger(cfg):
    """

    :param cfg:
    :return:
    """
    model_name = cfg.MODEL.MODEL_NAME
    dataset_name = cfg.DATASET.DATASET_NAME
    log_prefix_name = '{:s}_{:s}_classification_train'.format(dataset_name, model_name)
    return init_logger.get_logger(
        cfg=cfg,
        log_file_name_prefix=log_prefix_name
    )
