#! /usr/bin/python
# -*- coding: utf-8 -*-
"""You will learn.

1. What is TF-Slim ?
1. How to combine TensorLayer and TF-Slim ?

Introduction of Slim    : https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim
Slim Pre-trained Models : https://github.com/tensorflow/models/tree/master/research/slim

With the help of SlimNetsLayer, all Slim Model can be combined into TensorLayer.
All models in the following link, end with `return net, end_points`` are available.
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/slim/python/slim/nets

Bugs
-----
tf.variable_scope :
        https://groups.google.com/a/tensorflow.org/forum/#!topic/discuss/RoxrU3UnbFA
load inception_v3 for prediction:
        http://stackoverflow.com/questions/39357454/restore-checkpoint-in-tensorflow-tensor-name-not-found

"""

import os
import time
import numpy as np
import tensorflow as tf
#from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import (resnet_v2_50,resnet_v2_101,resnet_v2_152 resnet_arg_scope)
from tensorflow.contrib.slim.python.slim.nets.resnet_v1 import (resnet_v1_50,resnet_v1_101,resnet_v1_152, resnet_arg_scope)
import tensorlayer as tl

slim = tf.contrib.slim

model_name="resnet_v1_50"

model_=resnet_v1_50


def resnet(x,num_classes=1000,is_train=False,reuse=False):

    net_in = tl.layers.InputLayer(x, name='input_layer')
    with slim.arg_scope(resnet_arg_scope()):
        ## Alternatively, you should implement inception_v3 without TensorLayer as follow.
        # logits, end_points = inception_v3(X, num_classes=1011,
        #                                   is_training=False)
        network = tl.layers.SlimNetsLayer(
            prev_layer=net_in,
            slim_layer=model_,
            slim_args={
                'num_classes': num_classes,
                'is_training': is_train,
                'reuse':reuse

            },
            name=model_name  # <-- the name should be the same with the ckpt model
        )
        y = tf.reshape(network.outputs, [-1, num_classes]) 
#        y = tf.nn.softmax(y)
    return network,y
        






