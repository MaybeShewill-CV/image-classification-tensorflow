#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/18 下午7:12
# @Author  : LuoYao
# @Site    : ICode
# @File    : __init__.py.py
# @IDE: PyCharm
"""
dataset provider
"""
import importlib


def get_dataset_provider(cfg):
    """

    :param cfg:
    :return:
    """
    dataset_name = cfg.DATASET.DATASET_NAME
    module_name = '{:s}_reader'.format(dataset_name)
    mod = importlib.import_module(module_name)

    return getattr(mod, "get_provider")
