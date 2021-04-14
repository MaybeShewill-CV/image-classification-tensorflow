#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/29 上午11:04
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : evaluate_model.py
# @IDE: PyCharm
"""
evaluate model
"""
import argparse
import time
import os.path as ops

import tqdm
import tensorflow as tf
import cv2
import loguru
import numpy as np
from sklearn.metrics import (precision_score, recall_score,
                             precision_recall_curve, average_precision_score, f1_score)
from sklearn.metrics import classification_report
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.fixes import signature
import matplotlib.pyplot as plt

import cls_model_zoo
from local_utils import config_utils

LOG = loguru.logger


def init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, help='The network you used', default='xception')
    parser.add_argument('--dataset', type=str, help='The dataset', default='ilsvrc_2012')
    parser.add_argument('--weights_path', type=str, help='The ckpt weights file path')
    parser.add_argument('--dataset_flag', type=str, default='val')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--need_shuffle', type=args_str2bool, default=True)

    return parser.parse_args()


def args_str2bool(arg_value):
    """

    :param arg_value:
    :return:
    """
    if arg_value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    elif arg_value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def plot_precision_recall_curve(labels, predictions_prob, class_nums, average_function='weighted'):
    """
    Plot precision recall curve
    :param labels:
    :param predictions_prob:
    :param class_nums:
    :param average_function:
    :return:
    """
    labels = label_binarize(labels, classes=np.linspace(0, class_nums - 1, num=class_nums).tolist())
    predictions_prob = np.array(predictions_prob, dtype=np.float32)

    precision = dict()
    recall = dict()
    average_precision = dict()

    for i in range(class_nums):
        precision[i], recall[i], _ = precision_recall_curve(labels[:, i],
                                                            predictions_prob[:, i])
        average_precision[i] = average_precision_score(labels[:, i], predictions_prob[:, i])

    precision[average_function], recall[average_function], _ = precision_recall_curve(
        labels.ravel(), predictions_prob.ravel())
    average_precision[average_function] = average_precision_score(
        labels, predictions_prob, average=average_function)
    LOG.info('Average precision score, {:s}-averaged '
             'over all classes: {:.5f}'.format(average_function, average_precision[average_function]))

    plt.figure()
    plt.step(recall[average_function], precision[average_function], color='b', alpha=0.2,
             where='post')
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.fill_between(recall[average_function], precision[average_function], alpha=0.2, color='b',
                     **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, {:s}-averaged over '
        'all classes: AP={:.5f}'.format(average_function, average_precision[average_function]))


def calculate_evaluate_statics(labels, predictions, model_name='ilsvrc_2012_xception', avgerage_method='weighted'):
    """
    Calculate Precision, Recall and F1 score
    :param labels:
    :param predictions:
    :param model_name:
    :param avgerage_method:
    :return:
    """
    LOG.info('Model name: {:s}:'.format(model_name))
    LOG.info('\tPrecision: {:.5f}'.format(
        precision_score(
            y_true=labels, y_pred=predictions, average=avgerage_method)
    ))
    LOG.info('\tRecall: {:.5f}'.format(
        recall_score(y_true=labels, y_pred=predictions, average=avgerage_method)
    ))
    LOG.info('\tF1: {:.5f}\n'.format(
        f1_score(y_true=labels, y_pred=predictions, average=avgerage_method)
    ))


def evaluate():
    """
    eval
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
    dataset_flag = args.dataset_flag
    batch_size = args.batch_size
    need_shuffle = args.need_shuffle

    loguru.logger.add(
        './log/{:s}_{:s}_evaluate.log'.format(dataset_name, net_name),
        level='INFO',
        format="{time} {level} {message}",
        retention="10 days",
        rotation="1 week"
    )

    LOG.info('Eval model name: {:s}'.format(net_name))
    LOG.info('Eval model config path: {:s}'.format(config_file_path))
    LOG.info('Eval model weights path: {:s}'.format(args.weights_path))
    LOG.info('Eval dataset name: {:s}'.format(dataset_name))
    LOG.info('Eval dataset flag: {:s}'.format(dataset_flag))

    if dataset_flag == 'val':
        image_file_list = cfg.DATASET.VAL_FILE_LIST
    elif dataset_flag == 'test':
        image_file_list = cfg.DATASET.TEST_FILE_LIST
    elif dataset_flag == 'train':
        image_file_list = cfg.DATASET.TRAIN_FILE_LIST
    else:
        raise ValueError('Wrong dataset flag')

    # define input image pathas and gt labels
    image_count = 0
    input_image_paths = []
    gt_labels = []
    with open(image_file_list, 'r') as file:
        for line in file:
            info = line.rstrip('\r').rstrip('\n').split(' ')
            input_image_paths.append(info[0])
            gt_labels.append(int(info[1]))
            image_count += 1
    image_count = int((image_count // batch_size) * batch_size)
    LOG.info('Eval image counts: {:d}'.format(image_count))

    if need_shuffle:
        idx = np.random.permutation(len(input_image_paths))
        input_image_paths = np.array(input_image_paths)[idx][:image_count]
        gt_labels = np.array(gt_labels)[idx][:image_count]
    else:
        input_image_paths = input_image_paths[:image_count]
        gt_labels = gt_labels[:image_count]

    # define input tensor
    input_tensor = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, cfg.AUG.EVAL_CROP_SIZE[1], cfg.AUG.EVAL_CROP_SIZE[0], 3],
        name='input_tensor'
    )
    label_tensor = tf.placeholder(dtype=tf.int32, shape=[batch_size])

    # define net
    net = cls_model_zoo.get_model(cfg=cfg, phase='test')
    logits = net.inference(input_tensor=input_tensor, name=cfg.MODEL.MODEL_NAME, reuse=False)
    score = tf.nn.softmax(logits)
    prediction = tf.argmax(score, axis=1)
    train_top1_acc, train_top1_update = tf.metrics.accuracy(
        labels=label_tensor,
        predictions=prediction
    )

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            cfg.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    # define session
    sess = tf.Session()

    # run session
    with sess.as_default():

        sess.run(tf.local_variables_initializer())
        saver.restore(sess, args.weights_path)

        predicted_result = []
        predicted_score = []
        batch_counts = int(image_count / batch_size)
        pbar_input_batches = tqdm.tqdm(range(batch_counts))
        for index, batch_iter in enumerate(pbar_input_batches):
            image_batch_paths = input_image_paths[batch_iter * batch_size:batch_iter * batch_size + batch_size]
            input_gt_labels = gt_labels[batch_iter * batch_size:batch_iter * batch_size + batch_size]
            input_images = []
            for image_path in image_batch_paths:
                input_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                # input_image = cv2.blur(input_image, (3, 3))
                # input_image = cv2.GaussianBlur(input_image, (3, 3), 1.0)
                input_image = cv2.resize(input_image, tuple(cfg.AUG.EVAL_CROP_SIZE), interpolation=cv2.INTER_LINEAR)
                input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                input_image = input_image.astype(np.float32)
                input_image = input_image / 127.5 - 1.0
                input_images.append(input_image)

            t_start = time.time()
            _, predicted_labels, score_values = sess.run(
                fetches=[train_top1_update, prediction, score],
                feed_dict={
                    input_tensor: input_images,
                    label_tensor: input_gt_labels
                }
            )
            t_cost = time.time() - t_start
            predicted_result.extend(predicted_labels)
            predicted_score.extend(score_values)
            pbar_input_batches.set_description('Cost time: {:.4f}s'.format(t_cost))

        # print prediction report
        LOG.info(
            '{:s} {:s} classification_report(left: labels):'.format(cfg.DATASET.DATASET_NAME, cfg.MODEL.MODEL_NAME)
        )
        LOG.info(classification_report(gt_labels, predicted_result))

        # calculate evaluate statics
        calculate_evaluate_statics(labels=gt_labels, predictions=predicted_result, model_name='{:s}_{:s}'.format(
            cfg.DATASET.DATASET_NAME, cfg.MODEL.MODEL_NAME
        ))

        # plot precision recall curve
        enc = OneHotEncoder(handle_unknown='ignore')
        plot_precision_recall_curve(
            labels=enc.fit_transform(np.array(gt_labels).reshape((-1, 1))).toarray(),
            predictions_prob=predicted_score,
            class_nums=cfg.DATASET.NUM_CLASSES
        )
        plt.show()

    return


if __name__ == '__main__':
    """
    main
    """
    evaluate()
