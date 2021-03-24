#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/12/12 下午1:57
# @Author  : LuoYao
# @Site    : ICode
# @File    : augmentation_utils.py
# @IDE: PyCharm
"""
augmentation util function
"""
import cv2
import numpy as np


def _resize(img, cfg, mode='train'):
    """
    改变图像及标签图像尺寸
    AUG.AUG_METHOD为unpadding，所有模式均直接resize到AUG.FIX_RESIZE_SIZE的尺寸
    AUG.AUG_METHOD为stepscaling, 按比例resize，训练时比例范围AUG.MIN_SCALE_FACTOR到AUG.MAX_SCALE_FACTOR,
    间隔为AUG.SCALE_STEP_SIZE，其他模式返回原图
    AUG.AUG_METHOD为rangescaling，长边对齐，短边按比例变化，训练时长边对齐范围AUG.MIN_RESIZE_VALUE
    到AUG.MAX_RESIZE_VALUE，其他模式长边对齐AUG.INF_RESIZE_VALUE
    Args：
        img(numpy.ndarray): 输入图像
        grt(numpy.ndarray): 标签图像，默认为None
        mode(string): 模式, 默认训练模式
    Returns：
        resize后的图像和标签图
    """
    mode = mode.lower()
    if cfg.AUG.RESIZE_METHOD == 'unpadding':
        target_size = (cfg.AUG.FIX_RESIZE_SIZE[0], cfg.AUG.FIX_RESIZE_SIZE[1])
        if img is None:
            raise ValueError('Invalid image data')
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    elif cfg.AUG.RESIZE_METHOD == 'stepscaling':
        if mode == 'train':
            min_scale_factor = cfg.AUG.MIN_SCALE_FACTOR
            max_scale_factor = cfg.AUG.MAX_SCALE_FACTOR
            step_size = cfg.AUG.SCALE_STEP_SIZE
            scale_factor = _get_random_scale(
                min_scale_factor, max_scale_factor, step_size
            )
            img = _randomly_scale_image_and_label(
                img, scale=scale_factor
            )
    elif cfg.AUG.RESIZE_METHOD == 'rangescaling':
        min_resize_value = cfg.AUG.MIN_RESIZE_VALUE
        max_resize_value = cfg.AUG.MAX_RESIZE_VALUE
        if mode == 'train':
            if min_resize_value == max_resize_value:
                random_size = min_resize_value
            else:
                random_size = int(
                    np.random.uniform(min_resize_value, max_resize_value) + 0.5)
        else:
            random_size = cfg.AUG.INF_RESIZE_VALUE

        value = max(img.shape[0], img.shape[1])
        scale = float(random_size) / float(value)
        img = cv2.resize(
            img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
        )
    else:
        raise Exception("Unexpect data augmention method: {}".format(cfg.AUG.AUG_METHOD))

    return img


def _get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """
    在一定范围内得到随机值，范围为min_scale_factor到max_scale_factor，间隔为step_size
    Args：
        min_scale_factor(float): 随机尺度下限，大于0
        max_scale_factor(float): 随机尺度上限，不小于下限值
        step_size(float): 尺度间隔，非负, 等于为0时直接返回min_scale_factor到max_scale_factor范围内任一值
    Returns：
        随机尺度值
    """

    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return min_scale_factor

    if step_size == 0:
        return np.random.uniform(min_scale_factor, max_scale_factor)

    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = np.linspace(
        min_scale_factor, max_scale_factor, num_steps).tolist()
    np.random.shuffle(scale_factors)

    return scale_factors[0]


def _randomly_scale_image_and_label(image, scale=1.0):
    """
    按比例resize图像和标签图, 如果scale为1，返回原图
    Args：
        image(numpy.ndarray): 输入图像
        label(numpy.ndarray): 标签图，默认None
        sclae(float): 图片resize的比例，非负，默认1.0
    Returns：
        resize后的图像和标签图
    """

    if scale == 1.0:
        return image

    height = image.shape[0]
    width = image.shape[1]
    new_height = int(height * scale + 0.5)
    new_width = int(width * scale + 0.5)

    new_image = cv2.resize(
        image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return new_image


def _random_rotation(crop_img, rich_crop_max_rotation, mean_value):
    """
    随机旋转图像和标签图
    Args：
        crop_img(numpy.ndarray): 输入图像
        rich_crop_max_rotation(int)：旋转最大角度，0-90
        mean_value(list)：均值, 对图片旋转产生的多余区域使用均值填充
    Returns：
        旋转后的图像和标签图
    """
    if rich_crop_max_rotation > 0:
        (h, w) = crop_img.shape[:2]
        do_rotation = np.random.uniform(
            -rich_crop_max_rotation, rich_crop_max_rotation)
        pc = (w // 2, h // 2)
        r = cv2.getRotationMatrix2D(pc, do_rotation, 1.0)
        cos = np.abs(r[0, 0])
        sin = np.abs(r[0, 1])

        nw = int((h * sin) + (w * cos))
        nh = int((h * cos) + (w * sin))

        (cx, cy) = pc
        r[0, 2] += (nw / 2) - cx
        r[1, 2] += (nh / 2) - cy
        dsize = (nw, nh)
        crop_img = cv2.warpAffine(
            crop_img,
            r,
            dsize=dsize,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=mean_value
        )
    return crop_img


def _rand_scale_aspect(crop_img, rich_crop_min_scale=0, rich_crop_aspect_ratio=0):
    """
    从输入图像和标签图像中裁取随机宽高比的图像，并reszie回原始尺寸
    Args:
        crop_img(numpy.ndarray): 输入图像
        rich_crop_min_scale(float)：裁取图像占原始图像的面积比，0-1，默认0返回原图
        rich_crop_aspect_ratio(float): 裁取图像的宽高比范围，非负，默认0返回原图
    Returns:
        裁剪并resize回原始尺寸的图像和标签图像
    """
    if rich_crop_min_scale == 0 or rich_crop_aspect_ratio == 0:
        return crop_img
    else:
        img_height = crop_img.shape[0]
        img_width = crop_img.shape[1]
        for i in range(0, 10):
            area = img_height * img_width
            target_area = area * np.random.uniform(rich_crop_min_scale, 1.0)
            aspect_ratio = np.random.uniform(
                rich_crop_aspect_ratio, 1.0 / rich_crop_aspect_ratio
            )
            dw = int(np.sqrt(target_area * 1.0 * aspect_ratio))
            dh = int(np.sqrt(target_area * 1.0 / aspect_ratio))
            if np.random.randint(10) < 5:
                tmp = dw
                dw = dh
                dh = tmp
            if dh < img_height and dw < img_width:
                h1 = np.random.randint(0, img_height - dh)
                w1 = np.random.randint(0, img_width - dw)

                crop_img = crop_img[h1:(h1 + dh), w1:(w1 + dw), :]
                crop_img = cv2.resize(
                    crop_img, (img_width, img_height),
                    interpolation=cv2.INTER_LINEAR
                )
                break
    return crop_img


def _saturation_jitter(cv_img, jitter_range):
    """
    调节图像饱和度
    Args:
        cv_img(numpy.ndarray): 输入图像
        jitter_range(float): 调节程度，0-1
    Returns:
        饱和度调整后的图像
    """
    assert isinstance(cv_img, np.ndarray)
    grey_mat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    grey_mat = grey_mat[:, :, None] * np.ones(3, dtype=int)[None, None, :]
    cv_img = cv_img.astype(np.float32)
    cv_img = cv_img * (1 - jitter_range) + jitter_range * grey_mat
    cv_img = np.where(cv_img > 255, 255, cv_img)
    cv_img = cv_img.astype(np.uint8)

    return cv_img


def _brightness_jitter(cv_img, jitter_range):
    """
    调节图像亮度
    Args:
        cv_img(numpy.ndarray): 输入图像
        jitter_range(float): 调节程度，0-1
    Returns:
        亮度调整后的图像
    """
    assert isinstance(cv_img, np.ndarray)
    cv_img = cv_img.astype(np.float32)
    cv_img = cv_img * (1.0 - jitter_range)
    cv_img = np.where(cv_img > 255, 255, cv_img)
    cv_img = cv_img.astype(np.uint8)
    return cv_img


def _contrast_jitter(cv_img, jitter_range):
    """
    调节图像对比度
    Args:
        cv_img(numpy.ndarray): 输入图像
        jitter_range(float): 调节程度，0-1
    Returns:
        对比度调整后的图像
    """
    assert isinstance(cv_img, np.ndarray)
    grey_mat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(grey_mat)
    cv_img = cv_img.astype(np.float32)
    cv_img = cv_img * (1 - jitter_range) + jitter_range * mean
    cv_img = np.where(cv_img > 255, 255, cv_img)
    cv_img = cv_img.astype(np.uint8)
    return cv_img


def _random_jitter(cv_img, saturation_range, brightness_range, contrast_range):
    """
    图像亮度、饱和度、对比度调节，在调整范围内随机获得调节比例，并随机顺序叠加三种效果
    Args:
        cv_img(numpy.ndarray): 输入图像
        saturation_range(float): 饱和对调节范围，0-1
        brightness_range(float): 亮度调节范围，0-1
        contrast_range(float): 对比度调节范围，0-1
    Returns:
        亮度、饱和度、对比度调整后图像
    """

    saturation_ratio = np.random.uniform(-saturation_range, saturation_range)
    brightness_ratio = np.random.uniform(-brightness_range, brightness_range)
    contrast_ratio = np.random.uniform(-contrast_range, contrast_range)

    order = [1, 2, 3]
    np.random.shuffle(order)

    for i in range(3):
        if order[i] == 0:
            cv_img = _saturation_jitter(cv_img, saturation_ratio)
        if order[i] == 1:
            cv_img = _brightness_jitter(cv_img, brightness_ratio)
        if order[i] == 2:
            cv_img = _contrast_jitter(cv_img, contrast_ratio)
    return cv_img


def _hsv_color_jitter(crop_img, brightness_jitter_ratio=0, saturation_jitter_ratio=0, contrast_jitter_ratio=0):
    """
    图像亮度、饱和度、对比度调节
    Args:
        crop_img(numpy.ndarray): 输入图像
        brightness_jitter_ratio(float): 亮度调节度最大值，1-0，默认0
        saturation_jitter_ratio(float): 饱和度调节度最大值，1-0，默认0
        contrast_jitter_ratio(float): 对比度调节度最大值，1-0，默认0
    Returns：
        亮度、饱和度、对比度调节后图像
   """

    if brightness_jitter_ratio > 0 or \
        saturation_jitter_ratio > 0 or \
            contrast_jitter_ratio > 0:
        crop_img = _random_jitter(
            crop_img, saturation_jitter_ratio,
            brightness_jitter_ratio, contrast_jitter_ratio
        )
    return crop_img


def _rand_crop(crop_img, cfg, mode='train'):
    """
    随机裁剪图片和标签图, 若crop尺寸大于原始尺寸，分别使用均值和ignore值填充再进行crop，
    crop尺寸与原始尺寸一致，返回原图，crop尺寸小于原始尺寸直接crop
    Args:
        crop_img(numpy.ndarray): 输入图像
        cfg:
        mode(string): 模式, 默认训练模式，验证或预测、可视化模式时crop尺寸需大于原始图片尺寸
    Returns：
        裁剪后的图片和标签图
    """
    mode = mode.lower()
    img_height = crop_img.shape[0]
    img_width = crop_img.shape[1]

    if mode in ['train', 'validation']:
        crop_width = cfg.AUG.TRAIN_CROP_SIZE[0]
        crop_height = cfg.AUG.TRAIN_CROP_SIZE[1]
    else:
        crop_width = cfg.AUG.EVAL_CROP_SIZE[0]
        crop_height = cfg.AUG.EVAL_CROP_SIZE[1]

    if mode not in ['train', 'validation']:
        if crop_height < img_height or crop_width < img_width:
            raise Exception(
                "Crop size({},{}) must large than img size({},{}) when in EvalPhase."
                .format(crop_width, crop_height, img_width, img_height))

    if img_height == crop_height and img_width == crop_width:
        return crop_img
    else:
        pad_height = max(crop_height - img_height, 0)
        pad_width = max(crop_width - img_width, 0)
        if pad_height > 0 or pad_width > 0:
            crop_img = cv2.copyMakeBorder(
                crop_img, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=cfg.DATASET.PADDING_VALUE
            )
            img_height = crop_img.shape[0]
            img_width = crop_img.shape[1]

        if crop_height > 0 and crop_width > 0:
            h_off = np.random.randint(img_height - crop_height + 1)
            w_off = np.random.randint(img_width - crop_width + 1)

            crop_img = crop_img[h_off:(crop_height + h_off), w_off:(
                w_off + crop_width), :]
        return crop_img


def _rich_crop_image(img, cfg):
    """
    rich crop image
    :param img:
    :param cfg:
    :return:
    """
    if not cfg.AUG.RICH_CROP.ENABLE:
        return img
    # gaussian blur
    if cfg.AUG.RICH_CROP.BLUR:
        if cfg.AUG.RICH_CROP.BLUR_RATIO <= 0:
            n = 0
        elif cfg.AUG.RICH_CROP.BLUR_RATIO >= 1:
            n = 1
        else:
            n = int(1.0 / cfg.AUG.RICH_CROP.BLUR_RATIO)
        if n > 0:
            if np.random.randint(0, n) == 0:
                radius = np.random.randint(3, 10)
                if radius % 2 != 1:
                    radius = radius + 1
                if radius > 9:
                    radius = 9
                img = cv2.GaussianBlur(img, (radius, radius), 0, 0)
    # random rotation
    img = _random_rotation(
        img,
        rich_crop_max_rotation=cfg.AUG.RICH_CROP.MAX_ROTATION,
        mean_value=cfg.DATASET.PADDING_VALUE
    )
    # random scale
    img = _rand_scale_aspect(
        img,
        rich_crop_min_scale=cfg.AUG.RICH_CROP.MIN_AREA_RATIO,
        rich_crop_aspect_ratio=cfg.AUG.RICH_CROP.ASPECT_RATIO
    )
    # random hsv jitter
    img = _hsv_color_jitter(
        img,
        brightness_jitter_ratio=cfg.AUG.RICH_CROP.BRIGHTNESS_JITTER_RATIO,
        saturation_jitter_ratio=cfg.AUG.RICH_CROP.SATURATION_JITTER_RATIO,
        contrast_jitter_ratio=cfg.AUG.RICH_CROP.CONTRAST_JITTER_RATIO
    )
    return img


def _random_flip_image(img, cfg):
    """

    :param img:
    :param cfg:
    :return:
    """
    if cfg.AUG.FLIP:
        if cfg.AUG.FLIP_RATIO <= 0:
            n = 0
        elif cfg.AUG.FLIP_RATIO >= 1:
            n = 1
        else:
            n = int(1.0 / cfg.AUG.FLIP_RATIO)
        if n > 0:
            if np.random.randint(0, n) == 0:
                img = img[::-1, :, :]
    return img


def _random_mirror_image(img, cfg):
    """

    :param img:
    :param cfg:
    :return:
    """
    if cfg.AUG.MIRROR:
        if np.random.randint(0, 2) == 1:
            img = img[:, ::-1, :]
    return img


def _normalize_image(img):
    """

    :param img:
    :return:
    """
    img = img.astype(np.float32)
    # img = img - cfg.DATASET.MEAN_VALUE
    img = img / 127.5 - 1.0

    return img


def preprocess_image(src_image, cfg):
    """

    :param src_image:
    :param cfg:
    :return:
    """
    # resize image
    src_image = _resize(src_image, cfg=cfg)
    # random flip
    src_image = _random_flip_image(src_image, cfg=cfg)
    # random mirror
    src_image = _random_mirror_image(src_image, cfg=cfg)
    # rich crop
    src_image = _rich_crop_image(src_image, cfg=cfg)
    # random crop
    src_image = _rand_crop(src_image, cfg=cfg, mode='train')
    # normalize image
    src_image = _normalize_image(src_image)
    # cast image type
    src_image = src_image.astype(np.float32)

    return src_image
