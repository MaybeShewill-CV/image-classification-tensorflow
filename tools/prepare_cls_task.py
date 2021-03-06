#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2021/6/17 下午1:48
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : prepare_cls_task.py
# @IDE: PyCharm
"""
prepare classification task
"""
import argparse
import os
import os.path as ops
import yaml
import collections
import json
import glob

import loguru
import tqdm

from tools import make_dataset_tfrecords

LOG = loguru.logger
LOG.add(
    './log/prepare_cls_task.log',
    level='INFO',
)


def _check_src_image_dir_complete(task_dir):
    """

    :param task_dir:
    :return:
    """
    train_src_image_dir = ops.join(task_dir, 'src_image', 'train')
    if not ops.exists(train_src_image_dir):
        LOG.error('Source training image dir: {:s} not exist'.format(train_src_image_dir))
        return False

    test_src_image_dir = ops.join(task_dir, 'src_image', 'test')
    if not ops.exists(train_src_image_dir):
        LOG.error('Source testing image dir: {:s} not exist'.format(test_src_image_dir))
        return False

    val_src_image_dir = ops.join(task_dir, 'src_image', 'val')
    if not ops.exists(train_src_image_dir):
        LOG.error('Source val image dir: {:s} not exist'.format(val_src_image_dir))
        return False

    return True


def _check_image_classes_consistency(task_dir):
    """

    :param task_dir:
    :return:
    """
    if not _check_src_image_dir_complete(task_dir=task_dir):
        return False

    train_src_image_dir = ops.join(task_dir, 'src_image', 'train')
    train_cls_set = set(os.listdir(train_src_image_dir))

    test_src_image_dir = ops.join(task_dir, 'src_image', 'test')
    test_cls_set = set(os.listdir(test_src_image_dir))

    val_src_image_dir = ops.join(task_dir, 'src_image', 'val')
    val_cls_set = set(os.listdir(val_src_image_dir))

    if train_cls_set == test_cls_set == val_cls_set:
        return True
    else:
        LOG.error('Check image classes consistency failed. Train, '
                  'test, val dataset should share the same image classes, please check.')
        return False


def _generate_class_id_map_file(task_dir):
    """

    :param task_dir:
    :return:
    """
    if not _check_image_classes_consistency(task_dir=task_dir):
        LOG.error('Generate class id map file failed')
        return False

    train_src_image_dir = ops.join(task_dir, 'src_image', 'train')
    cls_list = os.listdir(train_src_image_dir)
    cls_map = collections.OrderedDict()
    for cls_id, cls_name in enumerate(cls_list):
        cls_map[cls_name] = cls_id

    with open(ops.join(task_dir, 'class_id_map.json'), 'w') as file:
        json.dump(cls_map, file)

    return True


def _make_image_index_file(task_dir):
    """

    :param task_dir:
    :return:
    """
    if not ops.exists(task_dir):
        LOG.error('Task dir: {:s} not exist'.format(task_dir))
        return False

    image_index_file_dir = ops.join(task_dir, 'image_file_index')
    os.makedirs(image_index_file_dir, exist_ok=True)

    cls_id_map_file = ops.join(task_dir, 'class_id_map.json')
    with open(cls_id_map_file, 'r') as file:
        cls_id_map = json.load(file)

    # generate train index file
    train_src_image_dir = ops.join(task_dir, 'src_image', 'train')
    train_src_image_paths = glob.glob('{:s}/**/*'.format(train_src_image_dir), recursive=True)
    train_src_image_paths = filter(
        lambda x: ops.split(x)[1].lower().endswith(('jpg', 'png', 'jpeg', 'tif', 'tiff')),
        train_src_image_paths
    )
    train_image_file_index_info = []
    train_image_paths_tqdm = tqdm.tqdm(train_src_image_paths)
    train_image_paths_tqdm.set_description('making train image index file')
    for img_path in train_image_paths_tqdm:
        folder_name = ops.split(ops.split(img_path)[0])[1]
        if folder_name not in cls_id_map:
            LOG.error('Class name: {:s} not existed in class id map file')
            return False
        cls_id = cls_id_map[folder_name]
        train_image_file_index_info.append('{:s} {:d}'.format(img_path, cls_id))

    # generate test index file
    test_src_image_dir = ops.join(task_dir, 'src_image', 'test')
    test_src_image_paths = glob.glob('{:s}/**/*'.format(test_src_image_dir), recursive=True)
    test_src_image_paths = filter(
        lambda x: ops.split(x)[1].lower().endswith(('jpg', 'png', 'jpeg', 'tif', 'tiff')),
        test_src_image_paths
    )
    test_image_file_index_info = []
    test_image_paths_tqdm = tqdm.tqdm(test_src_image_paths)
    test_image_paths_tqdm.set_description('making test image index file')
    for img_path in test_image_paths_tqdm:
        folder_name = ops.split(ops.split(img_path)[0])[1]
        if folder_name not in cls_id_map:
            LOG.error('Class name: {:s} not existed in class id map file')
            return False
        cls_id = cls_id_map[folder_name]
        test_image_file_index_info.append('{:s} {:d}'.format(img_path, cls_id))

    # generate val index file
    val_src_image_dir = ops.join(task_dir, 'src_image', 'val')
    val_src_image_paths = glob.glob('{:s}/**/*'.format(val_src_image_dir), recursive=True)
    val_src_image_paths = filter(
        lambda x: ops.split(x)[1].lower().endswith(('jpg', 'png', 'jpeg', 'tif', 'tiff')),
        val_src_image_paths
    )
    val_image_file_index_info = []
    val_image_paths_tqdm = tqdm.tqdm(val_src_image_paths)
    val_image_paths_tqdm.set_description('making val image index file')
    for img_path in val_image_paths_tqdm:
        folder_name = ops.split(ops.split(img_path)[0])[1]
        if folder_name not in cls_id_map:
            LOG.error('Class name: {:s} not existed in class id map file')
            return False
        cls_id = cls_id_map[folder_name]
        val_image_file_index_info.append('{:s} {:d}'.format(img_path, cls_id))

    with open(ops.join(image_index_file_dir, 'train.txt'), 'w') as file:
        file.write('\n'.join(train_image_file_index_info))
    with open(ops.join(image_index_file_dir, 'test.txt'), 'w') as file:
        file.write('\n'.join(test_image_file_index_info))
    with open(ops.join(image_index_file_dir, 'val.txt'), 'w') as file:
        file.write('\n'.join(val_image_file_index_info))

    LOG.info('Generate image index file complete')
    return True


def _generate_template_model_cfg_file(task_dir, net_name, dataset_name):
    """

    :param task_dir:
    :param net_name:
    :param dataset_name:
    :return:
    """
    if net_name not in ['darknet53', 'densenet', 'inceptionv4', 'mobilenetv2', 'res2net',
                        'resnet', 'se-resnet', 'vgg', 'xception']:
        raise NotImplementedError('Not support model: {:s}'.format(net_name))

    darknet53_model_cfg = {
        'MODEL_NAME': 'darknet53'
    }
    densenet_model_cfg = {
        'MODEL_NAME': 'densenet',
        'DENSENET': {
            'GROWTH_RATE': 32,
            'BLOCK_NUMS': 4,
            'ENABLE_BC': True,
            'BC_THETA': 0.5,
            'COMPOSITE_CHANNELS': 128,
            'NET_SIZE': 121,
        }
    }
    inceptionv4_model_cfg = {
        'MODEL_NAME': 'inceptionv4',
        'INCEPTION': {
            'REDUCTION_A_PARAMS': {
                'K': 192,
                'L': 224,
                'M': 256,
                'N': 384,
            }
        }
    }
    mobilenetv2_model_cfg = {
        'MODEL_NAME': 'mobilenetv2'
    }
    res2net_model_cfg = {
        'MODEL_NAME': 'res2net',
        'RES2NET': {
            'NET_SIZE': 50,
            'BOTTLENECK_SCALE': 4,
            'BOTTLENECK_BASE_WIDTH': 26,
        }
    }
    resnet_model_cfg = {
        'MODEL_NAME': 'resnet',
        'RESNET': {
            'NET_SIZE': 50
        }
    }
    seresnet_model_cfg = {
        'MODEL_NAME': 'se-resnet',
        'SE_RESNET': {
            'NET_SIZE': 50,
            'REDUCTION': 16
        }
    }
    vgg_model_cfg = {
        'MODEL_NAME': 'vgg',
        'VGG': {
            'NET_SIZE': 16
        }
    }
    xception_model_cfg = {
        'MODEL_NAME': 'xception',
        'XCEPTION': {
            'DEPTH_MULTIPLIER': 1,
            'LAYERS': 39
        }
    }

    if net_name == 'darknet53':
        model_cfg = darknet53_model_cfg
    elif net_name == 'densenet':
        model_cfg = densenet_model_cfg
    elif net_name == 'inceptionv4':
        model_cfg = inceptionv4_model_cfg
    elif net_name == 'mobilenetv2':
        model_cfg = mobilenetv2_model_cfg
    elif net_name == 'res2net':
        model_cfg = res2net_model_cfg
    elif net_name == 'resnet':
        model_cfg = resnet_model_cfg
    elif net_name == 'se-resnet':
        model_cfg = seresnet_model_cfg
    elif net_name == 'vgg':
        model_cfg = vgg_model_cfg
    elif net_name == 'xception':
        model_cfg = xception_model_cfg
    else:
        raise NotImplementedError('Not support model: {:s}'.format(net_name))
    aug_cfg = {
        'RESIZE_METHOD': 'unpadding',
        'FIX_RESIZE_SIZE': [256, 256],
        'INF_RESIZE_VALUE': 500,
        'MAX_RESIZE_VALUE': 600,
        'MIN_RESIZE_VALUE': 400,
        'MAX_SCALE_FACTOR': 2.0,
        'MIN_SCALE_FACTOR': 0.75,
        'SCALE_STEP_SIZE': 0.25,
        'TRAIN_CROP_SIZE': [224, 224],
        'EVAL_CROP_SIZE': [224, 224],
        'MIRROR': True,
        'FLIP': True,
        'FLIP_RATIO': 0.5,
        'RICH_CROP': {
            'ENABLE': False,
            'BLUR': True,
            'BLUR_RATIO': 0.2,
            'MAX_ROTATION': 15,
            'MIN_AREA_RATIO': 0.5,
            'ASPECT_RATIO': 0.5,
            'BRIGHTNESS_JITTER_RATIO': 0.5,
            'CONTRAST_JITTER_RATIO': 0.5,
        }
    }
    dataset_cfg = {
        'DATASET_NAME': '{:s}'.format(dataset_name),
        'DATA_DIR': '{:s}'.format(task_dir),
        'IMAGE_TYPE': 'rgb',
        'NUM_CLASSES': 1000,
        'TEST_FILE_LIST': '{:s}'.format(ops.join(task_dir, 'image_file_index', 'test.txt')),
        'TRAIN_FILE_LIST': '{:s}'.format(ops.join(task_dir, 'image_file_index', 'train.txt')),
        'VAL_FILE_LIST': '{:s}'.format(ops.join(task_dir, 'image_file_index', 'val.txt')),
        'IGNORE_INDEX': 255,
        'PADDING_VALUE': [0, 0, 0],
        'MEAN_VALUE': [123.68, 116.779, 103.939],
        'STD_VALUE': [58.393, 57.12, 57.375],
        'USE_ONE_HOT_LABEL': False
    }
    freeze_cfg = {
        'MODEL_FILENAME': 'model',
        'PARAMS_FILENAME': 'params'
    }
    test_cfg = {
        'TEST_MODEL': 'model/{:s}/final'.format(net_name)
    }
    train_cfg = {
        'MODEL_SAVE_DIR': 'model/{:s}_{:s}/'.format(net_name, dataset_name),
        'TBOARD_SAVE_DIR': 'tboard/{:s}_{:s}/'.format(net_name, dataset_name),
        'MODEL_PARAMS_CONFIG_FILE_NAME': "model_train_config.json",
        'RESTORE_FROM_SNAPSHOT': {
            'ENABLE': False,
            'SNAPSHOT_PATH': ''
        },
        'SNAPSHOT_EPOCH': 4,
        'BATCH_SIZE': 32,
        'EPOCH_NUMS': 129,
        'WARM_UP': {
            'ENABLE': True,
            'EPOCH_NUMS': 4
        },
        'FREEZE_BN': {
            'ENABLE': False
        },
        'USE_GENERAL_DATA_PROVIDER': {
            'ENABLE': True
        },
        'FAST_DATA_PROVIDER': {
            'ENABLE': True,
            'MULTI_PROCESSOR_NUMS': 4,
            'SHUFFLE_BUFFER_SIZE': 512,
            'PREFETCH_SIZE': 16
        },
        'DROPOUT': {
            'ENABLE': False,
            'KEEP_PROB': 0.7
        },
        'LABEL_SMOOTH': {
            'ENABLE': True,
            'SMOOTH_VALUE': 0.1
        }
    }
    solver_cfg = {
        'LR': 0.0125,
        'LR_POLICY': 'cos',
        'POLY_DECAY': {
            'LR_POLYNOMIAL_POWER': 0.95,
            'LR_POLYNOMIAL_END_LR': 0.000001
        },
        'EXP_DECAY': {
            'DECAY_RATE': 0.1,
            'APPLY_STAIRCASE': True
        },
        'COS_DECAY': {
            'ALPHA': 0.0
        },
        'PIECEWISE_DECAY': {
            'DECAY_RATE': 0.1,
            'DECAY_BOUNDARY': [30.0, 60.0, 90.0, 120.0]
        },
        'OPTIMIZER': 'sgd',
        'MOMENTUM': 0.9,
        'WEIGHT_DECAY': 0.0005,
        'MOVING_AVE_DECAY': 0.9995,
        'LOSS_TYPE': 'cross_entropy',
    }
    gpu_cfg = {
        'GPU_MEMORY_FRACTION': 0.9,
        'TF_ALLOW_GROWTH': True
    }
    log_cfg = {
        'SAVE_DIR': './log',
        'LEVEL': 'INFO'
    }

    cfg = dict()
    cfg['AUG'] = aug_cfg
    cfg['DATASET'] = dataset_cfg
    cfg['FREEZE'] = freeze_cfg
    cfg['MODEL'] = model_cfg
    cfg['TEST'] = test_cfg
    cfg['TRAIN'] = train_cfg
    cfg['SOLVER'] = solver_cfg
    cfg['GPU'] = gpu_cfg
    cfg['LOG'] = log_cfg

    config_file_name = '{:s}_{:s}.yaml'.format(dataset_name, net_name)
    config_file_path = ops.join('./config', config_file_name)
    with open(config_file_path, 'w') as file:
        yaml.dump(cfg, file)

    LOG.info('Generate template model training config file complete')

    return


def _args_str2bool(arg_value):
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


def _init_args():
    """

    :return:
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--task_dir', type=str, help='The source dataset dir')
    parser.add_argument('--net', type=str, help='Model name used for training')
    parser.add_argument('--dataset_name', type=str, help='The dataset name')
    parser.add_argument('--make_tfrecords', type=_args_str2bool, default=True, help='Whether to make tfrecords')

    return parser.parse_args()


def _prepare_classification_task(task_dir, net_name, dataset_name, make_tfrecords=True):
    """

    :param task_dir:
    :param net_name:
    :param dataset_name:
    :param make_tfrecords:
    :return:
    """
    # check if source image dir complete
    if not _check_src_image_dir_complete(task_dir=task_dir):
        LOG.error('Prepare image classification task: {:s}, failed'.format(task_dir))
        return

    # generate class map id file
    if not _generate_class_id_map_file(task_dir=task_dir):
        LOG.error('Prepare image classification task: {:s}, failed'.format(task_dir))
        return

    # generate image index file
    if not _make_image_index_file(task_dir=task_dir):
        LOG.error('Prepare image classification task: {:s}, failed'.format(task_dir))
        return

    # generate template model training config file
    _generate_template_model_cfg_file(
        task_dir=task_dir,
        net_name=net_name,
        dataset_name=dataset_name
    )

    if make_tfrecords:
        os.system('python ./tools/make_dataset_tfrecords.py --net {:s} --dataset {:s}'.format(net_name, dataset_name))

    return


def main():
    """
    main func
    :return:
    """
    args = _init_args()

    LOG.info('Start prepare image classification task')
    _prepare_classification_task(
        task_dir=args.task_dir,
        net_name=args.net,
        dataset_name=args.dataset_name,
        make_tfrecords=args.make_tfrecords
    )
    LOG.info('Complete prepare image classification task')

    return


if __name__ == '__main__':
    """
    main func
    """
    main()
