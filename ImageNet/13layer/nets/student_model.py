from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from nets import mylayers
import tensorlayer as tl
import tensorflow as tf

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import random_ops

way_of_weight_init = 'Glorot'
H = 'Glorot'
W1_init = tf.contrib.layers.xavier_initializer()
W_init = tf.random_uniform_initializer(-1,1)

def model(images,
          num_classes=1001,
          is_training=False,
          reuse=True,
          scope='modelB'):

  G = tf.get_default_graph()

  def fw(x):
    with G.gradient_override_map({"Sign": "Identity"}):
      return tf.sign(x)

  def hard_sigmoid(x):
      return tf.clip_by_value((x + 1.) / 2., 0, 1)

  def round(x):
    with G.gradient_override_map({"Round": "Identity"}):
      return tf.round(x)

  def fa(x):
    ab = hard_sigmoid(x)
    ab = round(ab)
    ab = ab * 2 - 1
    return ab

  with tf.variable_scope(scope, '13layer', [images, num_classes]):

    net = tl.layers.InputLayer(images, name='input')
    net = tl.layers.Conv2dLayer(
      net,
      shape=[7, 7, 3, 128],
      strides=[1, 2, 2, 1],
      padding='SAME',
      act=tf.identity,
      W_init=W1_init,
      b_init=None,
      name='1Convolution_Layer_Full_Precision'
    )
    net = tl.layers.PoolLayer(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                              padding='SAME', pool=tf.nn.max_pool, name='Pool_Layer1')
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='1Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='1bi_act')
    net = mylayers.Binarized_Convolution(
      net,
      shape=[3, 3, 128, 384],
      strides=[1, 1, 1, 1],
      padding='SAME',
      act=tf.identity,
      binarize_weight=fw,
      W_init=W_init,
      b_init=None,
      name='2Convolution_Layer_Binary'
    )
    net = tl.layers.PoolLayer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='VALID', pool=tf.nn.max_pool, name='Pool_Layer2')
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='2Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='2bi_act')
    net = mylayers.Binarized_Convolution(
      net,
      shape=[3, 3, 384, 384],
      strides=[1, 1, 1, 1],
      padding='SAME',
      act=tf.identity,
      binarize_weight=fw,
      W_init=W_init,
      b_init=None,
      name='3Convolution_Layer_Binary'
    )
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='3Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='3bi_act')
    net = mylayers.Binarized_Convolution(
      net,
      shape=[3, 3, 384, 384],
      strides=[1, 1, 1, 1],
      padding='SAME',
      act=tf.identity,
      binarize_weight=fw,
      W_init=W_init,
      b_init=None,
      name='4Convolution_Layer_Binary'
    )
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='4Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='4bi_act')
    net = mylayers.Binarized_Convolution(
      net,
      shape=[3, 3, 384, 384],
      strides=[1, 1, 1, 1],
      padding='SAME',
      act=tf.identity,
      binarize_weight=fw,
      W_init=W_init,
      b_init=None,
      name='5Convolution_Layer_Binary'
    )
    net = tl.layers.PoolLayer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='VALID', pool=tf.nn.max_pool, name='Pool_Layer5')
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='5Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='5bi_act')
    net = mylayers.Binarized_Convolution(
      net,
      shape=[3, 3, 384, 512],
      strides=[1, 1, 1, 1],
      padding='SAME',
      act=tf.identity,
      binarize_weight=fw,
      W_init=W_init,
      b_init=None,
      name='6Convolution_Layer_Binary'
    )
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='6Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='6bi_act')
    net = mylayers.Binarized_Convolution(
      net,
      shape=[3, 3, 512, 512],
      strides=[1, 1, 1, 1],
      padding='SAME',
      act=tf.identity,
      binarize_weight=fw,
      W_init=W_init,
      b_init=None,
      name='7Convolution_Layer_Binary'
    )
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='7Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='7bi_act')
    net = mylayers.Binarized_Convolution(
      net,
      shape=[3, 3, 512, 512],
      strides=[1, 1, 1, 1],
      padding='SAME',
      act=tf.identity,
      binarize_weight=fw,
      W_init=W_init,
      b_init=None,
      name='8Convolution_Layer_Binary'
    )
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='8Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='8bi_act')
    net = mylayers.Binarized_Convolution(
      net,
      shape=[3, 3, 512, 512],
      strides=[1, 1, 1, 1],
      padding='SAME',
      act=tf.identity,
      binarize_weight=fw,
      W_init=W_init,
      b_init=None,
      name='9Convolution_Layer_Binary'
    )
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='9Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='9bi_act')
    net = mylayers.Binarized_Convolution(
      net,
      shape=[3, 3, 512, 512],
      strides=[1, 1, 1, 1],
      padding='SAME',
      act=tf.identity,
      binarize_weight=fw,
      W_init=W_init,
      b_init=None,
      name='10Convolution_Layer_Binary'
    )
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='10Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='10bi_act')
    net = mylayers.Binarized_Convolution(
      net,
      shape=[3, 3, 512, 512],
      strides=[1, 1, 1, 1],
      padding='SAME',
      act=tf.identity,
      binarize_weight=fw,
      W_init=W_init,
      b_init=None,
      name='11Convolution_Layer_Binary'
    )
    net = tl.layers.PoolLayer(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='VALID', pool=tf.nn.max_pool, name='Pool_Layer11')
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='11Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='11bi_act')
    net = tl.layers.FlattenLayer(net, name='flatten_layer')
    net = mylayers.Binarized_DenseLayer(
      net,
      n_units=4096,
      act=tf.identity,
      binarized_weight=fw,
      W_init=W_init,
      b_init=None,
      name='1Fully_Connected_Layer_Binary'
    )
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='12Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='12bi_act')
    net = tl.layers.DenseLayer(
      net,
      n_units=num_classes,
      act=tf.identity,
      W_init=W1_init,
      b_init=tf.zeros_initializer(),
      name='Fully_Connected_Output_Layer_Full_Precision'
    )
  return net, net.outputs
