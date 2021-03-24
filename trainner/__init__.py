#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/24 下午3:16
# @Author  : LuoYao
# @Site    : ICode
# @File    : __init__.py.py
# @IDE: PyCharm
"""
model trainner
"""
from trainner import base_trainner
from local_utils import config_utils


def get_resnet_trainner(config_file_path):
    """

    :param config_file_path:
    :return:
    """
    cfg = config_utils.get_config(config_file_path=config_file_path)

    return base_trainner.BaseClsTrainner(cfg=cfg)
