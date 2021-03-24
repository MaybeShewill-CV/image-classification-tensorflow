#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/16 下午1:24
# @Author  : LuoYao
# @Site    : ICode
# @File    : __init__.py.py
# @IDE: PyCharm
"""
parse config file
"""
import local_utils.config_utils.parse_config_utils


def get_config(config_file_path):
    """

    :param config_file_path:
    :return:
    """
    return parse_config_utils.Config(config_path=config_file_path)
