#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2020/5/8 下午4:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/image-classification-tensorflow
# @File    : augmentation_tf_utils.py
# @IDE: PyCharm
"""
Tensorflow version image data augmentation tools
"""
import tensorflow as tf
import numpy as np

from local_utils import config_utils


def decode(serialized_example, cfg):
    """
    Parses an image and label from the given `serialized_example`
    :param serialized_example:
    :param cfg:
    :return:
    """
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'gt_src_image_raw': tf.FixedLenFeature([], tf.string),
            'gt_label_image_raw': tf.FixedLenFeature([], tf.string),
        })

    # decode gt image
    gt_image = tf.image.decode_png(features['gt_src_image_raw'], channels=3)
    gt_image = tf.reshape(gt_image, shape=[cfg.AUG.TRAIN_CROP_SIZE[1], cfg.AUG.TRAIN_CROP_SIZE[0], 3])

    # decode gt binary image
    gt_binary_image = tf.image.decode_png(features['gt_label_image_raw'], channels=1)
    gt_binary_image = tf.reshape(gt_binary_image, shape=[cfg.AUG.TRAIN_CROP_SIZE[1], cfg.AUG.TRAIN_CROP_SIZE[0], 1])

    return gt_image, gt_binary_image


def resize(img, cfg, mode='train', align_corners=True):
    """
    resize image
    :param img:
    :param cfg:
    :param mode:
    :param align_corners:
    :return:
    """
    mode = mode.lower()
    img = tf.expand_dims(img, axis=0)
    if cfg.AUG.RESIZE_METHOD == 'unpadding':
        target_size = (cfg.AUG.FIX_RESIZE_SIZE[1], cfg.AUG.FIX_RESIZE_SIZE[0])
        img = tf.image.resize_bilinear(images=img, size=target_size, align_corners=align_corners)
    elif cfg.AUG.RESIZE_METHOD == 'stepscaling':
        if mode == 'train':
            min_scale_factor = cfg.AUG.MIN_SCALE_FACTOR
            max_scale_factor = cfg.AUG.MAX_SCALE_FACTOR
            step_size = cfg.AUG.SCALE_STEP_SIZE
            scale_factor = get_random_scale(
                min_scale_factor, max_scale_factor, step_size
            )
            img = randomly_scale_image_and_label(
                img, scale=scale_factor, align_corners=align_corners
            )
    elif cfg.AUG.RESIZE_METHOD == 'rangescaling':
        min_resize_value = cfg.AUG.MIN_RESIZE_VALUE
        max_resize_value = cfg.AUG.MAX_RESIZE_VALUE
        if mode == 'train':
            if min_resize_value == max_resize_value:
                random_size = min_resize_value
            else:
                random_size = tf.random.uniform(
                    shape=[1], minval=min_resize_value, maxval=max_resize_value, dtype=tf.float32) + 0.5
                random_size = tf.cast(random_size, dtype=tf.int32)
        else:
            random_size = cfg.AUG.INF_RESIZE_VALUE

        value = tf.maximum(img.shape[0], img.shape[1])
        scale = float(random_size) / float(value)
        target_size = (int(img.shape[0] * scale), int(img.shape[1] * scale))
        img = tf.image.resize_bilinear(images=img, size=target_size, align_corners=align_corners)
    else:
        raise Exception("Unexpect data augmention method: {}".format(cfg.AUG.AUG_METHOD))

    return tf.squeeze(img, axis=0)


def _image_dimensions(image, rank):
    """
    Returns the dimensions of an image tensor.
    :param image:
    :param rank:
    :return:
    """
    if image.get_shape().is_fully_defined():
        return image.get_shape().as_list()
    else:
        static_shape = image.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(image), rank)
        return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]


def pad_to_bounding_box(
        image, offset_height, offset_width, target_height,
        target_width, pad_value):
    """

    :param image:
    :param offset_height:
    :param offset_width:
    :param target_height:
    :param target_width:
    :param pad_value:
    :return:
    """
    with tf.name_scope(None, 'pad_to_bounding_box', [image]):
        image = tf.convert_to_tensor(image, name='image')
        original_dtype = image.dtype
        if original_dtype != tf.float32 and original_dtype != tf.float64:
            image = tf.cast(image, tf.int32)
        image_rank_assert = tf.Assert(
            tf.logical_or(
                tf.equal(tf.rank(image), 3),
                tf.equal(tf.rank(image), 4)),
            ['Wrong image tensor rank.'])
        with tf.control_dependencies([image_rank_assert]):
            image -= pad_value
        image_shape = image.get_shape()
        is_batch = True
        if image_shape.ndims == 3:
            is_batch = False
            image = tf.expand_dims(image, 0)
        elif image_shape.ndims is None:
            is_batch = False
            image = tf.expand_dims(image, 0)
            image.set_shape([None] * 4)
        elif image.get_shape().ndims != 4:
            raise ValueError('Input image must have either 3 or 4 dimensions.')
        _, height, width, _ = _image_dimensions(image, rank=4)
        target_width_assert = tf.Assert(
            tf.greater_equal(
                target_width, width),
            ['target_width must be >= width'])
        target_height_assert = tf.Assert(
            tf.greater_equal(target_height, height),
            ['target_height must be >= height'])
        with tf.control_dependencies([target_width_assert]):
            after_padding_width = target_width - offset_width - width
        with tf.control_dependencies([target_height_assert]):
            after_padding_height = target_height - offset_height - height
        offset_assert = tf.Assert(
            tf.logical_and(
                tf.greater_equal(after_padding_width, 0),
                tf.greater_equal(after_padding_height, 0)),
            ['target size not possible with the given target offsets'])
        batch_params = tf.stack([0, 0])
        height_params = tf.stack([offset_height, after_padding_height])
        width_params = tf.stack([offset_width, after_padding_width])
        channel_params = tf.stack([0, 0])
        with tf.control_dependencies([offset_assert]):
            paddings = tf.stack([batch_params, height_params, width_params, channel_params])
        padded = tf.pad(image, paddings)
        if not is_batch:
            padded = tf.squeeze(padded, axis=[0])
        outputs = padded + pad_value
        if outputs.dtype != original_dtype:
            outputs = tf.cast(outputs, original_dtype)
        return outputs


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """
    get random scale
    :param min_scale_factor:
    :param max_scale_factor:
    :param step_size:
    :return:
    """

    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return tf.cast(min_scale_factor, tf.float32)

        # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return tf.random_uniform([1],
                                 minval=min_scale_factor,
                                 maxval=max_scale_factor)

        # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = tf.lin_space(min_scale_factor, max_scale_factor, num_steps)
    shuffled_scale_factors = tf.random_shuffle(scale_factors)
    return shuffled_scale_factors[0]


def randomly_scale_image_and_label(image, scale=1.0, align_corners=True):
    """

    :param image:
    :param scale:
    :param align_corners:
    :return:
    """
    if scale == 1.0:
        return image
    image_shape = tf.shape(image)
    new_dim = tf.cast(
        tf.cast([image_shape[1], image_shape[2]], tf.float32) * scale,
        tf.int32)

    image = tf.image.resize_bilinear(image, new_dim, align_corners=align_corners)
    return image


def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """

    :param image:
    :param offset_height:
    :param offset_width:
    :param crop_height:
    :param crop_width:
    :return:
    """
    original_shape = tf.shape(image)

    if len(image.get_shape().as_list()) != 3:
        raise ValueError('input must have rank of 3')
    original_channels = image.get_shape().as_list()[2]

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.cast(tf.stack([offset_height, offset_width, 0]), tf.int32)

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    image = tf.reshape(image, cropped_shape)
    image.set_shape([crop_height, crop_width, original_channels])
    return image


def rand_crop(image_list, crop_height, crop_width):
    """

    :param image_list:
    :param crop_height:
    :param crop_width:
    :return:
    """
    if not image_list:
        raise ValueError('Empty image_list.')

        # Compute the rank assertions.
    rank_assertions = []
    for i in range(len(image_list)):
        image_rank = tf.rank(image_list[i])
        rank_assert = tf.Assert(
            tf.equal(image_rank, 3),
            ['Wrong rank for tensor  %s [expected] [actual]',
             image_list[i].name, 3, image_rank])
        rank_assertions.append(rank_assert)

    with tf.control_dependencies([rank_assertions[0]]):
        image_shape = tf.shape(image_list[0])
    image_height = image_shape[0]
    image_width = image_shape[1]
    crop_size_assert = tf.Assert(
        tf.logical_and(
            tf.greater_equal(image_height, crop_height),
            tf.greater_equal(image_width, crop_width)),
        ['Crop size greater than the image size.'])

    asserts = [rank_assertions[0], crop_size_assert]

    for i in range(1, len(image_list)):
        image = image_list[i]
        asserts.append(rank_assertions[i])
        with tf.control_dependencies([rank_assertions[i]]):
            shape = tf.shape(image)
        height = shape[0]
        width = shape[1]

        height_assert = tf.Assert(
            tf.equal(height, image_height),
            ['Wrong height for tensor %s [expected][actual]',
             image.name, height, image_height])
        width_assert = tf.Assert(
            tf.equal(width, image_width),
            ['Wrong width for tensor %s [expected][actual]',
             image.name, width, image_width])
        asserts.extend([height_assert, width_assert])

    with tf.control_dependencies(asserts):
        max_offset_height = tf.reshape(image_height - crop_height + 1, [])
        max_offset_width = tf.reshape(image_width - crop_width + 1, [])
    offset_height = tf.random_uniform(
        [], maxval=max_offset_height, dtype=tf.int32)
    offset_width = tf.random_uniform(
        [], maxval=max_offset_width, dtype=tf.int32)

    return [_crop(image, offset_height, offset_width, crop_height, crop_width) for image in image_list]


def resolve_shape(tensor, rank=None, scope=None):
    """
    resolves the shape of a Tensor.
    :param tensor:
    :param rank:
    :param scope:
    :return:
    """
    with tf.name_scope(scope, 'resolve_shape', [tensor]):
        if rank is not None:
            shape = tensor.get_shape().with_rank(rank).as_list()
        else:
            shape = tensor.get_shape().as_list()

        if None in shape:
            shape_dynamic = tf.shape(tensor)
            for i in range(len(shape)):
                if shape[i] is None:
                    shape[i] = shape_dynamic[i]

    return shape


def random_flip_image(img, cfg):
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
            random_value = tf.random_uniform([])
            is_flipped = tf.less_equal(random_value, 0.5)
            img = tf.cond(is_flipped, true_fn=lambda: img[::-1, :, :], false_fn=lambda: img)

    return img


def random_mirror_image(img,  cfg):
    """

    :param img:
    :param cfg:
    :return:
    """
    if cfg.AUG.MIRROR:
        random_value = tf.random_uniform([])
        is_mirrored = tf.less_equal(random_value, 0.5)
        img = tf.cond(is_mirrored, true_fn=lambda: img[:, ::-1, :], false_fn=lambda: img)

    return img


def normalize_image(img, cfg=None):
    """

    :param img:
    :param cfg:
    :return:
    """
    # img = tf.divide(img, tf.constant(127.5)) - 1.0
    img_mean = tf.convert_to_tensor(
        np.array(cfg.DATASET.MEAN_VALUE).reshape((1, 1, len(cfg.DATASET.MEAN_VALUE))),
        dtype=tf.float32
    )
    img_std = tf.convert_to_tensor(
        np.array(cfg.DATASET.STD_VALUE).reshape((1, 1, len(cfg.DATASET.STD_VALUE))),
        dtype=tf.float32
    )
    img -= img_mean
    img /= img_std

    return img


def preprocess_image(src_image, cfg):
    """

    :param src_image:
    :param cfg:
    :return:
    """
    # resize image
    # src_image = resize(src_image, cfg)
    # random flip
    src_image = random_flip_image(src_image, cfg=cfg)
    # random mirror
    src_image = random_mirror_image(src_image, cfg=cfg)
    # random crop
    src_image = rand_crop(
        image_list=[src_image],
        crop_height=cfg.AUG.TRAIN_CROP_SIZE[1],
        crop_width=cfg.AUG.TRAIN_CROP_SIZE[0]
    )[0]
    # normalize image
    src_image = normalize_image(src_image, cfg=cfg)

    return src_image


def test():
    """

    :return:
    """
    source_image_path = './CITYSPACES/gt_images/leftImg8bit/train/aachen/aachen_000000_000019_leftImg8bit.png'
    source_label_path = './CITYSPACES/gt_annotation/gtFine/train/aachen/aachen_000000_000019_gtFine_labelTrainIds.png'

    source_image = tf.io.read_file(source_image_path)
    source_image = tf.image.decode_png(source_image)
    source_image.set_shape(shape=[1024, 2048, 3])
    source_label_image = tf.io.read_file(source_label_path)
    source_label_image = tf.image.decode_png(source_label_image, channels=0)
    source_label_image.set_shape(shape=[1024, 2048, 1])

    preprocess_src_img = preprocess_image(
        src_image=source_image,
        cfg=config_utils.get_config('./config/ilsvrc_2012_resnet.yaml')
    )

    with tf.Session() as sess:
        while True:
            ret = sess.run([preprocess_src_img])
            print(ret[0].shape)


if __name__ == '__main__':
    """
    test code
    """
    test()
