#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/25 下午2:01
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : loss.py
# @IDE: PyCharm
"""
loss function
"""
import tensorflow as tf


def dice_bce_loss(logits, label_tensor, weight_decay, l2_vars, **kwargs):
    """

    :param logits:
    :param label_tensor:
    :param weight_decay:
    :param l2_vars:
    :return:
    """

    def __dice_loss(_y_pred, _y_true):
        """

        :param _y_pred:
        :param _y_true:
        :return:
        """
        _intersection = tf.reduce_sum(_y_true * _y_pred, axis=-1)
        _l = tf.reduce_sum(_y_pred * _y_pred, axis=-1)
        _r = tf.reduce_sum(_y_true * _y_true, axis=-1)
        _dice = (2.0 * _intersection + 1e-5) / (_l + _r + 1e-5)
        _dice = tf.reduce_mean(_dice)
        return 1.0 - _dice

    if 'class_nums' not in kwargs:
        raise ValueError('dice bce loss need class_nums params')
    class_nums = kwargs['class_nums']
    # compute dice loss
    local_label_tensor = tf.one_hot(label_tensor, depth=class_nums, dtype=tf.float32)
    principal_loss_dice = __dice_loss(tf.nn.softmax(logits), local_label_tensor)
    principal_loss_dice = tf.identity(principal_loss_dice, name='principal_loss_dice')

    # compute bce loss
    principal_loss_bce = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_tensor, logits=logits)
    )
    principal_loss_bce = tf.identity(principal_loss_bce, name='principal_loss_bce')

    # compute l2 loss
    l2_reg_loss = tf.constant(0.0, tf.float32)
    for vv in l2_vars:
        if 'beta' in vv.name or 'gamma' in vv.name or 'bias' in vv.name:
            continue
        else:
            l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
    l2_reg_loss *= weight_decay
    l2_reg_loss = tf.identity(l2_reg_loss, 'l2_loss')
    total_loss = principal_loss_dice + principal_loss_bce + l2_reg_loss
    total_loss = tf.identity(total_loss, name='total_loss')

    ret = {
        'total_loss': total_loss,
        'principal_loss': principal_loss_bce + principal_loss_dice,
        'l2_loss': l2_reg_loss,
    }

    return ret


def cross_entropy_loss(logits, label_tensor, weight_decay, l2_vars, **kwargs):
    """

    :param logits:
    :param label_tensor:
    :param weight_decay:
    :param l2_vars:
    :param kwargs:
    :return:
    """
    if 'use_label_smooth' not in kwargs:
        use_label_smooth = False
    else:
        use_label_smooth = kwargs['use_label_smooth']
    if use_label_smooth and 'lb_smooth_value' not in kwargs:
        lb_smooth_value = 0.1
    elif use_label_smooth:
        lb_smooth_value = kwargs['lb_smooth_value']
    else:
        lb_smooth_value = 0.0

    if use_label_smooth:
        labels = tf.one_hot(
            indices=label_tensor,
            depth=kwargs['class_nums'],
            on_value=1.0 - lb_smooth_value,
            off_value=lb_smooth_value / kwargs['class_nums'],
            dtype=tf.float32
        )
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=labels,
            logits=logits,
            name='cross_entropy_per_example'
        )
        print('Label Smooth')
    else:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_tensor,
            name='cross_entropy_per_example'
        )
        print('Not Label Smooth')

    cross_entropy_loss_value = tf.reduce_mean(cross_entropy, name='cross_entropy')
    l2_reg_loss = tf.constant(0.0, tf.float32)
    for vv in l2_vars:
        if 'beta' in vv.name or 'gamma' in vv.name or 'bias' in vv.name:
            continue
        else:
            l2_reg_loss = tf.add(l2_reg_loss, tf.nn.l2_loss(vv))
    l2_reg_loss *= weight_decay
    l2_reg_loss = tf.identity(l2_reg_loss, 'l2_loss')

    total_loss = cross_entropy_loss_value + l2_reg_loss
    total_loss = tf.identity(total_loss, 'total_loss')

    ret = {
        'total_loss': total_loss,
        'principal_loss': cross_entropy_loss_value,
        'l2_loss': l2_reg_loss
    }
    return ret
