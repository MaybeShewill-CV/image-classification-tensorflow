#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/24 下午1:36
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : __init__.py
# @IDE: PyCharm
"""
cls model zoom
"""
import os.path as ops
import importlib


def get_model(cfg, phase):
    """
    Fetch Network Function Pointer
    """
    model_name = cfg.MODEL.MODEL_NAME
    module_dir_name = ops.dirname(__file__)
    module_dir_name = ops.split(module_dir_name)[-1]
    module_name = '{:s}.{:s}'.format(module_dir_name, model_name)
    mod = importlib.import_module(module_name)

    return getattr(mod, "get_model")(cfg=cfg, phase=phase)
