#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/24 下午1:36
# @Author  : LuoYao
# @Site    : ICode
# @File    : __init__.py.py
# @IDE: PyCharm
"""
cls model zoom
"""
import importlib


def get_model(cfg):
    """
    Fetch Network Function Pointer
    """
    model_name = cfg.MODEL.MODEL_NAME
    mod = importlib.import_module(model_name)

    return getattr(mod, "get_model")
