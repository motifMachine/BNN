#! /usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
blocks_per_group=4   #l
widening_factor =10  #k
w_init=tf.variance_scaling_initializer()
b_init = tf.constant_initializer(value=0.0)
b1_init=None
#b1_init = tf.constant_initializer(value=0.0)
def zero_pad_channels(x, pad=0):
    """
    Function for Lambda layer
    """
    pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
    return tf.pad(x, pattern)

def residual_block(x, count, nb_filters=16, subsample_factor=1,is_train=False,dropout=True):
    prev_nb_channels = x.outputs.get_shape().as_list()[3]

    if subsample_factor > 1:
        subsample = [1, subsample_factor, subsample_factor, 1]
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
        # shortcut: identity
        shortcut = x

    if nb_filters > prev_nb_channels:
        name_lambda = 'lambda_layer' + str(count)
        shortcut = tl.layers.LambdaLayer(
            shortcut,
            zero_pad_channels,
            fn_args={'pad': nb_filters - prev_nb_channels},
            name=name_lambda)

    name_norm = 'norm' + str(count)
    y = tl.layers.BatchNormLayer(x,
                                 decay=0.999,
                                 epsilon=1e-05,
                                 is_train=is_train,
                                 act=tf.nn.relu,
                                 name=name_norm)

    name_conv = 'conv_layer' + str(count)
    y = tl.layers.Conv2dLayer(y,
                              
                              shape=[3, 3, prev_nb_channels, nb_filters],
                              strides=subsample,
                              padding='SAME',
                              W_init=w_init,
                              b_init=b1_init,
                              name=name_conv)

    name_norm_2 = 'norm_second' + str(count)
    y = tl.layers.BatchNormLayer(y,
                                 decay=0.999,
                                 epsilon=1e-05,
                                 is_train=is_train,
                                 act=tf.nn.relu,
                                 name=name_norm_2)
    if dropout:                             
        y = tl.layers.DropoutLayer(y, keep=0.7, is_fix=True,
                        is_train=is_train, name='drop'+str(count))
                        
    prev_input_channels = y.outputs.get_shape().as_list()[3]
    name_conv_2 = 'conv_layer_second' + str(count)
    y = tl.layers.Conv2dLayer(y,
                              
                              shape=[3, 3, prev_input_channels, nb_filters],
                              strides=[1, 1, 1, 1],
                              padding='SAME',
                              W_init=w_init,
                              b_init=b1_init,
                              name=name_conv_2)

    name_merge = 'merge' + str(count)
    out = tl.layers.ElementwiseLayer([y, shortcut],
                                     combine_fn=tf.add,
                                     name=name_merge)


    return out

    
def wresnet_T(x,num_classes=10,is_train=False,reuse=tf.AUTO_REUSE):
    with tf.device('/gpu'):
        with tf.variable_scope("wresnet_T", reuse=reuse):
            net = tl.layers.InputLayer(x, name='input_layer')
            net = tl.layers.Conv2dLayer(net,
                                      shape=[3, 3, 3, 64],
                                      strides=[1, 1, 1, 1],
                                      padding='SAME',
                                      W_init=w_init,
                                      b_init=b1_init,
                                      name='cnn_layer_first')

            for i in range(0, blocks_per_group):
                nb_filters = 16 * widening_factor
                count = i
                net = residual_block(net, count, nb_filters=nb_filters, subsample_factor=1,is_train=is_train)

            for i in range(0, blocks_per_group):
                nb_filters = 32 * widening_factor
                if i == 0:
                    subsample_factor = 2
                else:
                    subsample_factor = 1
                count = i + blocks_per_group
                net = residual_block(net, count, nb_filters=nb_filters, subsample_factor=subsample_factor,is_train=is_train)

            for i in range(0, blocks_per_group):
                nb_filters = 64 * widening_factor
                if i == 0:
                    subsample_factor = 2
                else:
                    subsample_factor = 1
                count = i + 2*blocks_per_group
                net = residual_block(net, count, nb_filters=nb_filters, subsample_factor=subsample_factor,is_train=is_train)

            net = tl.layers.BatchNormLayer(net,
                                         decay=0.999,
                                         epsilon=1e-05,
                                         is_train=is_train,
                                         act=tf.nn.relu,
                                         name='norm_last')

            net = tl.layers.PoolLayer(net,
                                    ksize=[1, 8, 8, 1],
                                    strides=[1, 8, 8, 1],
                                    padding='VALID',
                                    pool=tf.nn.avg_pool,
                                    name='pool_last')

            net = tl.layers.FlattenLayer(net, name='flatten')

            net = tl.layers.DenseLayer(net,
                                     W_init=w_init,
                                     b_init=b_init,
                                     n_units=num_classes,
                                     act=tf.identity,
                                     name='fc')

            y = net.outputs
            
            return net,y