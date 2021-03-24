#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/24 下午3:16
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : __init__.py.py
# @IDE: PyCharm
"""
model trainner
"""
from trainner import base_trainner
from local_utils import config_utils


def get_ilsvrc_resnet_trainner(config_file_path='./config/ilsvrc_2012_resnet.yaml'):
    """

    :param config_file_path:
    :return:
    """
    cfg = config_utils.get_config(config_file_path=config_file_path)

    return base_trainner.BaseClsTrainner(cfg=cfg)


def get_ilsvrc_xception_trainner(config_file_path='./config/ilsvrc_2012_xception.yaml'):
    """

    :param config_file_path:
    :return:
    """
    cfg = config_utils.get_config(config_file_path=config_file_path)

    return base_trainner.BaseClsTrainner(cfg=cfg)


def get_ilsvrc_mobilenetv2_trainner(config_file_path='./config/ilsvrc_2012_mobilenetv2.yaml'):
    """

    :param config_file_path:
    :return:
    """
    cfg = config_utils.get_config(config_file_path=config_file_path)

    return base_trainner.BaseClsTrainner(cfg=cfg)


def get_trainner(cfg):
    """

    :param cfg:
    :return:
    """

    return base_trainner.BaseClsTrainner(cfg=cfg)
