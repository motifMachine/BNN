#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
from nets import mylayers
blocks_per_group=2   #l
widening_factor =1  #k
_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5
W1_init=tf.variance_scaling_initializer()

W_init=tf.random_uniform_initializer(-1,1)
b_init = tf.constant_initializer(value=0.0)
#b1_init = tf.constant_initializer(value=0.0)
b1_init=None
def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)

def residual_block(x, count, nb_filters=16, subsample_factor=1,is_train=False,dropout=False):
    prev_nb_channels = x.outputs.get_shape().as_list()[3]

    if subsample_factor > 1:
        subsample = [1, subsample_factor, subsample_factor, 1]
        conv_subs=(subsample_factor,subsample_factor)
        # shortcut: subsample + zero-pad channel dim
        name_pool = 'pool_layer' + str(count)
        shortcut = tl.layers.PoolLayer(x,
                                       ksize=subsample,
                                       strides=subsample,
                                       padding='VALID',
                                       pool=tf.nn.avg_pool,
                                       name=name_pool)

    else:
        subsample = [1, 1, 1, 1]
        conv_subs=(1,1)
        # shortcut: identity
        shortcut = x

    if nb_filters > prev_nb_channels:
        name_lambda = 'lambda_layer' + str(count)
        shortcut = tl.layers.LambdaLayer(
            shortcut,
            zero_pad_channels,
            fn_args={'pad': nb_filters - prev_nb_channels},
            name=name_lambda)

    name_norm = 'cnn_norm' + str(count)
    y = tl.layers.BatchNormLayer(x,
                                 decay=_BATCH_NORM_DECAY,
                                 epsilon=_BATCH_NORM_EPSILON,
                                 is_train=is_train,
                                 name=name_norm)
    y=mylayers.SignLayer(y)
    name_conv = str(count)+'conv_layer_bnn' 
    # y = tl.layers.BinaryConv2d(y,nb_filters,(3,3),
    #                           strides=conv_subs,
    #                           padding='SAME',
    #                           W_init=W_init,
    #                           b_init=b1_init,
    #                           name=name_conv)

    y = mylayers.BinaryConv2d(y,nb_filters,(3,3),
                              strides=conv_subs,
                              padding='SAME',
                              W_init=W_init,
                              b_init=b1_init,
                              name=name_conv)

    name_norm_2 = 'cnn_norm_second' + str(count)
    y = tl.layers.BatchNormLayer(y,
                                 decay=_BATCH_NORM_DECAY,
                                 epsilon=_BATCH_NORM_EPSILON,
                                 is_train=is_train,
                                 name=name_norm_2)
    if dropout:                             
        y = tl.layers.DropoutLayer(y, keep=0.7, is_fix=True,
                        is_train=is_train, name='drop'+str(count))
    y=mylayers.SignLayer(y)                    
    name_conv_2 = str(count)+'conv_layer_second_bnn'
        
    y = mylayers.BinaryConv2d(y,nb_filters,(3,3),
                              strides=(1,1),
                              padding='SAME',
                              W_init=W_init,
                              b_init=b1_init,
                              name=name_conv_2)

    name_merge = 'merge' + str(count)
    out = tl.layers.ElementwiseLayer([y, shortcut],
                                     combine_fn=tf.add,
                                     name=name_merge)


    return out

    
def wresnet(x,num_classes=1000,is_train=False,reuse=tf.AUTO_REUSE):

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

    with tf.device('/gpu:0'):
        with tf.variable_scope("wresnet", reuse=reuse):
            net = tl.layers.InputLayer(x, name='input_layer')
            net = tl.layers.Conv2dLayer(net,
                                      shape=[7, 7, 3, 64],
                                      strides=[1, 2, 2, 1],
                                      padding='SAME',
                                      W_init=W1_init,
                                      b_init=b1_init,
                                      name='cnn_layer_first')
            net = tl.layers.PoolLayer(net,
                                    ksize=[1, 3, 3, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME',
                                    pool=tf.nn.max_pool,
                                    name='pool_first')
                                    
            for i in range(0, blocks_per_group):
                nb_filters = 64 * widening_factor
                count = i
                net = residual_block(net, count, nb_filters=nb_filters, subsample_factor=1,is_train=is_train)

            for i in range(0, blocks_per_group):
                nb_filters = 128 * widening_factor
                if i == 0:
                    subsample_factor = 2
                else:
                    subsample_factor = 1
                count = i + blocks_per_group
                net = residual_block(net, count, nb_filters=nb_filters, subsample_factor=subsample_factor,is_train=is_train)
                
            for i in range(0, blocks_per_group):
                nb_filters = 256 * widening_factor
                if i == 0:
                    subsample_factor = 2
                else:
                    subsample_factor = 1
                count = i + 2*blocks_per_group
                net = residual_block(net, count, nb_filters=nb_filters, subsample_factor=subsample_factor,is_train=is_train)
                
            for i in range(0, blocks_per_group):
                nb_filters = 512 * widening_factor
                if i == 0:
                    subsample_factor = 2
                else:
                    subsample_factor = 1
                count = i + 3*blocks_per_group
                net = residual_block(net, count, nb_filters=nb_filters, subsample_factor=subsample_factor,is_train=is_train)

            net = tl.layers.BatchNormLayer(net,
                                         decay=_BATCH_NORM_DECAY,
                                         epsilon=_BATCH_NORM_EPSILON,
                                         is_train=is_train,
                                         name='cnn_norm_last')
            #net=tl.layers.SignLayer(net)
            net = tl.layers.PoolLayer(net,
                                    ksize=[1, 7, 7, 1],
                                    strides=[1, 7, 7, 1],
                                    padding='VALID',
                                    pool=tf.nn.avg_pool,
                                    name='pool_last')

            net = tl.layers.FlattenLayer(net, name='flatten')

            net = tl.layers.DenseLayer(net,
                                     W_init=W1_init,
                                     b_init=b_init,
                                     n_units=num_classes,
                                     act=tf.identity,
                                     name='cnn_fc')

            y = net.outputs
            
            return net,y
