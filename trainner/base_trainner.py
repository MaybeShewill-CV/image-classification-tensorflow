#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/3/24 下午3:28
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : base_trainner.py
# @IDE: PyCharm
"""
Trainner for image classification
"""
import os
import os.path as ops
import shutil
import time
import math

import numpy as np
import tensorflow as tf
import loguru
import tqdm

import cls_model_zoo
import data_provider

LOG = loguru.logger


class BaseClsTrainner(object):
    """
    init base trainner
    """

    def __init__(self, cfg):
        """
        initialize base trainner
        """
        # define solver params and dataset
        self._cfg = cfg
        self._model_name = self._cfg.MODEL.MODEL_NAME
        self._dataset_name = self._cfg.DATASET.DATASET_NAME
        LOG.info('Start initializing {:s} trainner for {:s}'.format(self._model_name, self._dataset_name))

        self._dataset_reader = data_provider.get_dataset_provider(cfg=cfg)
        self._train_dataset = self._dataset_reader.train_dataset
        self._val_dataset = self._dataset_reader.val_dataset
        self._steps_per_epoch = len(self._train_dataset)
        self._init_learning_rate = self._cfg.SOLVER.LR
        self._moving_ave_decay = self._cfg.SOLVER.MOVING_AVE_DECAY
        self._train_epoch_nums = self._cfg.TRAIN.EPOCH_NUMS
        self._batch_size = self._cfg.TRAIN.BATCH_SIZE
        self._momentum = self._cfg.SOLVER.MOMENTUM
        self._model_save_dir = self._cfg.TRAIN.MODEL_SAVE_DIR
        self._snapshot_epoch = self._cfg.TRAIN.SNAPSHOT_EPOCH
        self._input_tensor_size = self._cfg.AUG.FIX_RESIZE_SIZE
        self._lr_polynimal_decay_power = self._cfg.SOLVER.LR_POLYNOMIAL_POWER
        self._optimizer_mode = self._cfg.SOLVER.OPTIMIZER.lower()
        self._tboard_save_dir = self._cfg.TRAIN.TBOARD_SAVE_DIR
        if self._cfg.TRAIN.RESTORE_FROM_SNAPSHOT.ENABLE:
            self._initial_weight = self._cfg.TRAIN.RESTORE_FROM_SNAPSHOT.SNAPSHOT_PATH
        else:
            self._initial_weight = None
        if self._cfg.TRAIN.WARM_UP.ENABLE:
            self._warmup_epoches = self._cfg.TRAIN.WARM_UP.EPOCH_NUMS
        else:
            self._warmup_epoches = 0

        # define tensorflow session
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        sess_config.gpu_options.per_process_gpu_memory_fraction = self._cfg.GPU.GPU_MEMORY_FRACTION
        sess_config.gpu_options.allow_growth = self._cfg.GPU.TF_ALLOW_GROWTH
        sess_config.gpu_options.allocator_type = 'BFC'
        self._sess = tf.Session(config=sess_config)

        # define graph input tensor
        with tf.variable_scope(name_or_scope='graph_input_node'):
            self._input_src_image = tf.placeholder(
                dtype=tf.float32,
                shape=[self._batch_size, self._input_tensor_size[1], self._input_tensor_size[0], 3],
                name='inpue_source_image'
            )
            self._input_label = tf.placeholder(
                dtype=tf.int64,
                shape=[self._batch_size],
                name='input_label_image'
            )

        # define model loss
        self._model = cls_model_zoo.get_model(cfg=cfg, phase='train')
        loss_set = self._model.compute_loss(
            input_tensor=self._input_src_image,
            label=self._input_label,
            name=self._model_name,
            reuse=False
        )
        self._loss = loss_set['total_loss']
        self._cross_entropy_loss = loss_set['principal_loss']
        self._l2_loss = loss_set['l2_loss']
        self._logits = self._model.inference(
            input_tensor=self._input_src_image,
            name=self._model_name,
            reuse=True
        )
        self._prediciton = tf.nn.softmax(self._logits)
        self._prediciton = tf.argmax(self._prediciton, axis=1, name='{:s}_prediction'.format(self._model_name))

        # define top k accuracy
        with tf.variable_scope('accuracy'):
            self._prediction_acc, self._acc_update = tf.metrics.accuracy(
                labels=self._input_label,
                predictions=self._prediciton,
                name='train_metric_accuracy'
            )
            # self._cls_acc, self._cls_acc_update = tf.metrics.mean_per_class_accuracy(
            #     labels=self._input_label,
            #     predictions=self._prediciton,
            #     num_classes=CFG.DATASET.CLASS_NUMS
            # )
            self._val_prediction_acc, self._val_acc_update = tf.metrics.accuracy(
                labels=self._input_label,
                predictions=self._prediciton,
                name='validation_metric_accuracy'
            )

        # define learning rate
        with tf.variable_scope('learning_rate'):
            self._global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='global_step')
            self._val_global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='val_global_step')
            warmup_steps = tf.constant(
                self._warmup_epoches * self._steps_per_epoch, dtype=tf.float32, name='warmup_steps'
            )
            train_steps = tf.constant(
                self._train_epoch_nums * self._steps_per_epoch, dtype=tf.float32, name='train_steps'
            )
            self._learn_rate = tf.cond(
                pred=self._global_step < warmup_steps,
                true_fn=lambda: self._global_step / warmup_steps * self._init_learning_rate,
                false_fn=lambda: tf.train.polynomial_decay(
                    learning_rate=self._init_learning_rate,
                    global_step=self._global_step,
                    decay_steps=train_steps,
                    end_learning_rate=0.000001,
                    power=self._lr_polynimal_decay_power
                )
            )
            global_step_update = tf.assign_add(self._global_step, 1.0)
            val_global_step_update = tf.assign_add(self._val_global_step, 1.0)

        # define moving average op
        with tf.variable_scope(name_or_scope='moving_avg'):
            if self._cfg.TRAIN.FREEZE_BN.ENABLE:
                train_var_list = [
                    v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
                ]
            else:
                train_var_list = tf.trainable_variables()
            moving_ave_op = tf.train.ExponentialMovingAverage(
                self._moving_ave_decay).apply(train_var_list + tf.moving_average_variables())

        # define training op
        with tf.variable_scope(name_or_scope='train_step'):
            if self._cfg.TRAIN.FREEZE_BN.ENABLE:
                train_var_list = [
                    v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name
                ]
            else:
                train_var_list = tf.trainable_variables()
            if self._optimizer_mode == 'sgd':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate=self._learn_rate,
                    momentum=self._momentum
                )
            elif self._optimizer_mode == 'adam':
                optimizer = tf.train.AdamOptimizer(
                    learning_rate=self._learn_rate,
                )
            else:
                raise ValueError('Not support optimizer: {:s}'.format(self._optimizer_mode))
            optimize_op = optimizer.minimize(self._loss, var_list=train_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([optimize_op, global_step_update]):
                    with tf.control_dependencies([moving_ave_op]):
                        self._train_op = tf.no_op()

        # define saver and loader
        with tf.variable_scope('loader_and_saver'):
            self._net_var = tf.global_variables()
            self._loader = tf.train.Saver(self._net_var)
            self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # define summary
        with tf.variable_scope('summary'):
            lr_summary = tf.summary.scalar("learn_rate", self._learn_rate)
            total_loss_summary = tf.summary.scalar("total_loss", self._loss)
            cross_entropy_loss_summary = tf.summary.scalar("principal_loss", self._cross_entropy_loss)
            l2_loss = tf.summary.scalar('l2_loss', self._l2_loss)
            with tf.control_dependencies([self._acc_update]):
                train_acc_summary = tf.summary.scalar('train_prediction_accuracy', self._prediction_acc)
            with tf.control_dependencies([self._val_acc_update, val_global_step_update]):
                val_acc_summary = tf.summary.scalar('validation_prediction_accuracy', self._val_prediction_acc)
            tboard_dir = self._tboard_save_dir
            if ops.exists(tboard_dir):
                shutil.rmtree(tboard_dir)
            os.makedirs(tboard_dir, exist_ok=True)
            model_params_file_save_path = ops.join(self._tboard_save_dir, self._cfg.TRAIN.MODEL_PARAMS_CONFIG_FILE_NAME)
            with open(model_params_file_save_path, 'w', encoding='utf-8') as f_obj:
                self._cfg.dump_to_json_file(f_obj)
            self._write_summary_op = tf.summary.merge(
                [lr_summary, total_loss_summary, cross_entropy_loss_summary, l2_loss, train_acc_summary]
            )
            self._validation_summary_op = tf.summary.merge([val_acc_summary])
            self._summary_writer = tf.summary.FileWriter(tboard_dir, graph=self._sess.graph)

        LOG.info('Initialize {:s} {:s} trainner complete'.format(self._dataset_name, self._model_name))

    def train(self):
        """

        :return:
        """
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        if self._cfg.TRAIN.RESTORE_FROM_SNAPSHOT.ENABLE:
            try:
                LOG.info('=> Restoring weights from: {:s} ... '.format(self._initial_weight))
                self._loader.restore(self._sess, self._initial_weight)
                global_step_value = self._sess.run(self._global_step)
                remain_epoch_nums = self._train_epoch_nums - math.floor(global_step_value / self._steps_per_epoch)
                epoch_start_pt = self._train_epoch_nums - remain_epoch_nums
            except OSError as e:
                LOG.error(e)
                LOG.info('=> {:s} does not exist !!!'.format(self._initial_weight))
                LOG.info('=> Now it starts to train {:s} from scratch ...'.format(self._model_name))
                epoch_start_pt = 1
            except Exception as e:
                LOG.error(e)
                LOG.info('=> Can not load pretrained model weights: {:s}'.format(self._initial_weight))
                LOG.info('=> Now it starts to train {:s} from scratch ...'.format(self._model_name))
                epoch_start_pt = 1
        else:
            LOG.info('=> Starts to train {:s} from scratch ...'.format(self._model_name))
            epoch_start_pt = 1

        for epoch in range(epoch_start_pt, self._train_epoch_nums):
            traindataset_pbar = tqdm.tqdm(self._train_dataset)
            valdataset_pbar = tqdm.tqdm(self._val_dataset)
            train_epoch_losses = []
            train_epoch_acc = []
            # train_epoch_cls_acc = []
            val_epoch_losses = []
            val_epoch_acc = []

            for samples in traindataset_pbar:
                source_imgs = samples[0]
                labels = samples[1]
                if source_imgs is None or labels is None:
                    continue

                _, summary, train_step_loss, acc_value, global_step_value = self._sess.run(
                    fetches=[self._train_op, self._write_summary_op,
                             self._loss, self._prediction_acc, self._global_step],
                    feed_dict={
                        self._input_src_image: source_imgs,
                        self._input_label: labels,
                    }
                )
                train_epoch_losses.append(train_step_loss)
                train_epoch_acc.append(acc_value)
                # train_epoch_cls_acc.append(cls_acc_value)
                self._summary_writer.add_summary(summary, global_step=global_step_value)
                traindataset_pbar.set_description(
                        'train loss: {:.5f}, train acc: {:.5f}'.format(train_step_loss, acc_value)
                )

            for samples in valdataset_pbar:
                source_imgs = samples[0]
                labels = samples[1]
                if source_imgs is None or labels is None:
                    continue

                val_summary, val_step_loss, val_acc, val_global_step_value = self._sess.run(
                    [self._validation_summary_op, self._loss, self._val_prediction_acc, self._val_global_step],
                    feed_dict={
                        self._input_src_image: source_imgs,
                        self._input_label: labels,
                    }
                )
                val_epoch_losses.append(val_step_loss)
                val_epoch_acc.append(val_acc)
                self._summary_writer.add_summary(val_summary, global_step=val_global_step_value)
                valdataset_pbar.set_description('test acc: {:.5f}'.format(val_acc))

            train_epoch_losses = np.mean(train_epoch_losses)
            val_epoch_losses = np.mean(val_epoch_losses)
            train_epoch_acc = np.mean(train_epoch_acc)
            val_epoch_acc = np.mean(val_epoch_acc)

            if epoch % self._snapshot_epoch == 0:
                snapshot_model_name = '{:s}_val_acc={:.4f}.ckpt'.format(self._model_name, val_epoch_acc)
                snapshot_model_path = ops.join(self._model_save_dir, snapshot_model_name)
                os.makedirs(self._model_save_dir, exist_ok=True)
                self._saver.save(self._sess, snapshot_model_path, global_step=epoch)

            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            LOG.info(
                '=> Epoch: {:d} Time: {:s} Train loss: {:.5f} Test loss: {:.5f} '
                'Train acc: {:.5f} Test acc: {:.5f} ...'.format(
                    epoch, log_time, train_epoch_losses, val_epoch_losses,
                    train_epoch_acc, val_epoch_acc
                )
            )
        LOG.info('Complete training process good luck!!')

        return

    @property
    def dataset_provider(self):
        """

        :return:
        """
        return self._dataset_reader

    @property
    def model(self):
        """

        :return:
        """
        return self._model
