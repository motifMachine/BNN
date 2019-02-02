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

def modelB(images,
          num_classes=1000,
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

  with tf.variable_scope(scope, 'BNN_cifar10', [images, num_classes]):

    net = tl.layers.InputLayer(images, name='input')
    net = tl.layers.Conv2dLayer(
      net,
      shape=[11, 11, 3, 96],
      strides=[1, 4, 4, 1],
      padding='VALID',
      act=tf.identity,
      W_init=W1_init,
      b_init=None,
      name='1Convolution_Layer_Full_Precision'
    )
    net = tl.layers.PoolLayer(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                              padding='VALID', pool=tf.nn.max_pool, name='Pool_Layer1')
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='1Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='1bi_act')
    net = mylayers.Binarized_Convolution(
      net,
      shape=[5, 5, 96, 256],
      strides=[1, 1, 1, 1],
      padding='SAME',
      act=tf.identity,
      binarize_weight=fw,
      W_init=W_init,
      b_init=None,
      name='2Convolution_Layer_Binary'
    )
    net = tl.layers.PoolLayer(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                              padding='VALID', pool=tf.nn.max_pool, name='Pool_Layer2')
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='2Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='2bi_act')
    net = mylayers.Binarized_Convolution(
      net,
      shape=[3, 3, 256, 384],
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
      shape=[3, 3, 384, 256],
      strides=[1, 1, 1, 1],
      padding='SAME',
      act=tf.identity,
      binarize_weight=fw,
      W_init=W_init,
      b_init=None,
      name='5Convolution_Layer_Binary'
    )
    net = tl.layers.PoolLayer(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                              padding='VALID', pool=tf.nn.max_pool, name='Pool_Layer5')
    net = tl.layers.BatchNormLayer(net, act=tf.identity,epsilon=0.0001,
                                   is_train=is_training, name='5Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='5bi_act')
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
                                   is_train=is_training, name='6Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='6bi_act')
    net = mylayers.Binarized_DenseLayer(
      net,
      n_units=4096,
      act=tf.identity,
      binarized_weight=fw,
      W_init=W_init,
      b_init=None,
      name='2Fully_Connected_Layer_Binary'
    )
    net = tl.layers.BatchNormLayer(net, act=tf.identity, epsilon=0.0001,
                                   is_train=is_training, name='7Batch_Norm')
    net = mylayers.BinaryLayer(net, act=fa, name='7bi_act')
    net = tl.layers.DenseLayer(
      net,
      n_units=num_classes,
      act=tf.identity,
      W_init=W1_init,
      b_init=tf.zeros_initializer(),
      name='Fully_Connected_Output_Layer_Full_Precision'
    )
    # net = mylayers.Binarized_DenseLayer(
    #   net,
    #   n_units=num_classes,
    #   act=tf.identity,
    #   binarized_weight=fw,
    #   W_init=W_init,
    #   b_init=tf.zeros_initializer(),
    #   name='Fully_Connected_Output_Layer_Binary'
    # )
  return net.outputs, net
