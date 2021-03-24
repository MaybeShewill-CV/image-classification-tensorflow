import os
import os.path as ops
import random
import glob

SOURCE_DATASET_DIR = './data/ilsvrc_2012_dataset'

source_image_dir = ops.join(SOURCE_DATASET_DIR, 'source_image')
image_file_index_dir = ops.join(SOURCE_DATASET_DIR, 'image_file_index')

assert ops.exists(source_image_dir)
os.makedirs(image_file_index_dir, exist_ok=True)

train_sample_info = []
train_souce_image_dir = ops.join(source_image_dir, 'train')
assert ops.exists(train_souce_image_dir)

for class_id in os.listdir(train_souce_image_dir):
    sample_dir = ops.join(train_souce_image_dir, class_id)
    if not ops.isdir(sample_dir):
        continue
    for source_image_path in glob.glob('{:s}/*JPEG'.format(sample_dir)):
        train_sample_info.append('{:s} {:d}'.format(source_image_path, int(class_id)))
random.shuffle(train_sample_info)

test_sample_info = []
test_souce_image_dir = ops.join(source_image_dir, 'test')
assert ops.exists(test_souce_image_dir)

for class_id in os.listdir(test_souce_image_dir):
    sample_dir = ops.join(test_souce_image_dir, class_id)
    if not ops.isdir(sample_dir):
        continue
    for source_image_path in glob.glob('{:s}/*JPEG'.format(sample_dir)):
        test_sample_info.append('{:s} {:d}'.format(source_image_path, int(class_id)))
random.shuffle(test_sample_info)

val_sample_info = []
val_souce_image_dir = ops.join(source_image_dir, 'val')
assert ops.exists(val_souce_image_dir)

for class_id in os.listdir(val_souce_image_dir):
    sample_dir = ops.join(val_souce_image_dir, class_id)
    if not ops.isdir(sample_dir):
        continue
    for source_image_path in glob.glob('{:s}/*JPEG'.format(sample_dir)):
        val_sample_info.append('{:s} {:d}'.format(source_image_path, int(class_id)))
random.shuffle(val_sample_info)

with open(ops.join(image_file_index_dir, 'train.txt'), 'w') as file:
    file.write('\n'.join(train_sample_info))

with open(ops.join(image_file_index_dir, 'test.txt'), 'w') as file:
    file.write('\n'.join(test_sample_info))

with open(ops.join(image_file_index_dir, 'val.txt'), 'w') as file:
    file.write('\n'.join(val_sample_info))
