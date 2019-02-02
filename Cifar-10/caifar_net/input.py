#! /usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import os.path
import numpy as np

_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS=3
length=16 #cifar10 =16 cifar100 =8


def Cutout (img):
  mask = np.ones((_HEIGHT, _WIDTH,_NUM_CHANNELS), np.float32)
  y = np.random.randint(_HEIGHT)
  x = np.random.randint(_WIDTH)
  y1 = np.clip(y - length // 2, 0, _HEIGHT)
  y2 = np.clip(y + length // 2, 0, _HEIGHT)
  x1 = np.clip(x - length // 2, 0, _WIDTH)
  x2 = np.clip(x + length // 2, 0, _WIDTH)
  mask[y1: y2, x1: x2,:] = 0.
  img=tf.multiply(img,mask)
  return img
def read_data(path, file, is_train=False,cutout=False):
  filename = os.path.join(path, file)
  print('-------------------data_dir:' + filename)
  filename_queue = tf.train.string_input_producer([filename])
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(serialized_example,
                                     features={
                                       'label': tf.FixedLenFeature([], tf.int64),
                                       'img_raw': tf.FixedLenFeature([], tf.string),
                                     })
  # You can do more image distortion here for training data
  img = tf.decode_raw(features['img_raw'], tf.float32)
  img = tf.reshape(img, [_HEIGHT, _WIDTH, _NUM_CHANNELS])
  
  if is_train ==True:
    # 1. Randomly crop a [height, width] section of the image.
    img = tf.image.resize_image_with_crop_or_pad(
        img, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    img = tf.random_crop(img, [_HEIGHT, _WIDTH, _NUM_CHANNELS])


    # 2. Randomly flip the image horizontally.
    img = tf.image.random_flip_left_right(img)
    
    # 3. Randomly change brightness.
    img = tf.image.random_brightness(img, max_delta=63)

    # 4. Randomly change contrast.
    img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
    img = tf.image.per_image_standardization(img)
    if cutout:
      img=Cutout(img)
    
  elif is_train == False:
      # 1. Crop the central [height, width] of the image.
      img = tf.image.resize_image_with_crop_or_pad(img, _HEIGHT, _WIDTH)  
      img = tf.image.per_image_standardization(img)

  
  label = tf.cast(features['label'], tf.int32)
  return img, label