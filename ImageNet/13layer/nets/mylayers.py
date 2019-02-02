from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Layer

class Binarized_DenseLayer(Layer):
    def __init__(
        self,
        layer = None,
        n_units = 100,
        act = tf.identity,
        binarized_weight = tf.identity,
        W_init = tf.truncated_normal_initializer(stddev=0.1),
        b_init = tf.constant_initializer(value=0.0),
        W_init_args = {},
        b_init_args = {},
        name ='b_dense_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units
        print("  [TL] Binarized_DenseLayer  %s: %d %s" % (self.name, self.n_units, act.__name__))
        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W', shape=(n_in, n_units), initializer=W_init, **W_init_args )
            bin_w = binarized_weight(W)
            if b_init is not None:
                try:
                    b = tf.get_variable(name='b', shape=(n_units), initializer=b_init, **b_init_args )
                except: # If initializer is a constant, do not specify shape.
                    b = tf.get_variable(name='b', initializer=b_init, **b_init_args )
                self.outputs = act(tf.matmul(self.inputs, bin_w) + b)
            else:
                self.outputs = act(tf.matmul(self.inputs, bin_w))

        # Hint : list(), dict() is pass by value (shallow), without them, it is
        # pass by reference.
        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        if b_init is not None:
            self.all_params.extend( [W, b] )
        else:
            self.all_params.extend( [W] )


class Binarized_Convolution(Layer):
    def __init__(
        self,
        layer=None,
        shape=[5, 5, 1, 100],
        strides=[1, 1, 1, 1],
        padding='SAME',
        act=tf.identity,
        binarize_weight=tf.identity,
        W_init=tf.truncated_normal_initializer(stddev=0.02),
        b_init=tf.zeros_initializer,
        W_init_args={},
        b_init_args={},
        use_cudnn_on_gpu=True,
        data_format=None,
        name='cnn_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] Binarized_Convolution %s: shape:%s strides:%s pad:%s activation:%s" %
              (self.name, str(shape), str(strides), padding, act.__name__))

        with tf.variable_scope(name) as vs:
            W = tf.get_variable(name='W_conv2d', shape=shape, initializer=W_init, **W_init_args)
            # print(tf.shape(W))
            bin_w = binarize_weight(W)
            if b_init:
                b = tf.get_variable(name='b_conv2d', shape=(shape[-1]), initializer=b_init, **b_init_args)
                self.outputs = act(tf.nn.conv2d(self.inputs, bin_w, strides=strides, padding=padding,
                                            use_cudnn_on_gpu=use_cudnn_on_gpu,
                                            data_format=data_format) + b)
            else:
                self.outputs = act(tf.nn.conv2d(self.inputs, bin_w, strides=strides, padding=padding,
                                            use_cudnn_on_gpu=use_cudnn_on_gpu,
                                            data_format=data_format))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])

class BinaryLayer(Layer):

    def __init__(
            self,
            layer=None,
            act=tf.identity,
            name="biary_layer"):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs

        self.outputs = act(self.inputs)

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)

        self.all_layers.extend([self.outputs])

class BatchNormLayer(Layer):
    """
    The :class:`BatchNormLayer` class is a normalization layer, see ``tf.nn.batch_normalization`` and ``tf.nn.moments``.

    Batch normalization on fully-connected or convolutional maps.

    Parameters
    -----------
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    decay : float, default is 0.9.
        A decay factor for ExponentialMovingAverage, use larger value for large dataset.
    epsilon : float
        A small float number to avoid dividing by 0.
    act : activation function.
    is_train : boolean
        Whether train or inference.
    beta_init : beta initializer
        The initializer for initializing beta
    gamma_init : gamma initializer
        The initializer for initializing gamma
    name : a string or None
        An optional name to attach to this layer.

    References
    ----------
    - `Source <https://github.com/ry/tensorflow-resnet/blob/master/resnet.py>`_
    - `stackoverflow <http://stackoverflow.com/questions/38312668/how-does-one-do-inference-with-batch-normalization-with-tensor-flow>`_
    """
    def __init__(
        self,
        layer = None,
        decay = 0.9,
        epsilon = 0.00001,
        act = tf.identity,
        is_train = False,
        beta_init = tf.zeros_initializer,
        gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002), # tf.ones_initializer,
        name ='batchnorm_layer',
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] BatchNormLayer %s: decay:%f epsilon:%f act:%s is_train:%s" %
                            (self.name, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[1]

        from tensorflow.python.training import moving_averages
        from tensorflow.python.ops import control_flow_ops

        with tf.variable_scope(name) as vs:
            axis = [0, 2, 3]

            ## 1. beta, gamma
            if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
                beta_init = beta_init()
            beta = tf.get_variable('beta', shape=params_shape,
                               initializer=beta_init,
                               trainable=is_train)#, restore=restore)

            gamma = tf.get_variable('gamma', shape=params_shape,
                                initializer=gamma_init, trainable=is_train,
                                )#restore=restore)

            ## 2.
            if tf.__version__ > '0.12.1':
                moving_mean_init = tf.zeros_initializer()
            else:
                moving_mean_init = tf.zeros_initializer
            moving_mean = tf.get_variable('moving_mean',
                                      params_shape,
                                      initializer=moving_mean_init,
                                      trainable=False,)#   restore=restore)
            moving_variance = tf.get_variable('moving_variance',
                                          params_shape,
                                          initializer=tf.constant_initializer(1.),
                                          trainable=False,)#   restore=restore)

            ## 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            try:    # TF12
                update_moving_mean = moving_averages.assign_moving_average(
                                moving_mean, mean, decay, zero_debias=False)     # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                                moving_variance, variance, decay, zero_debias=False) # if zero_debias=True, has bias
                # print("TF12 moving")
            except Exception as e:  # TF11
                update_moving_mean = moving_averages.assign_moving_average(
                                moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(
                                moving_variance, variance, decay)
                # print("TF11 moving")

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)

            if is_train:
                mean, var = mean_var_with_update()
                self.outputs = act( tf.nn.batch_normalization(self.inputs, mean, var, beta, gamma, epsilon) )
            else:
                self.outputs = act( tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon) )

            variables = [beta, gamma, moving_mean, moving_variance]

            # print(len(variables))
            # for idx, v in enumerate(variables):
            #     print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v))
            # exit()

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( variables )

def fixed_padding(inputs, kernel_size):
    """Pads the input along the spatial dimensions independently of input size.
    Args:
        inputs: A tensor of size [batch, channels, height_in, width_in] or
        [batch, height_in, width_in, channels] depending on data_format.
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                 S  hould be a positive integer.
        data_format: The input format ('channels_last' or 'channels_first').
    Returns:
        A tensor with the same format as the input with the data either intact
        (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg

    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0], [pad_beg, pad_end],
                                [pad_beg, pad_end]])
    return padded_inputs

class conv2d_fixed_padding(Layer):
    def __init__(
        self,
        layer=None,
        filters=2,
        kernel_size=1,
        strides=2,
        act=tf.identity,
        W_init=tf.contrib.layers.xavier_initializer_conv2d(),
        b_init=None,
        W_init_args={},
        b_init_args={},
        use_cudnn_on_gpu=True,
        data_format='NCHW',
        name='conv2d_fixed_padding',
    ):
        if strides ==1:
            padding = 'SAME'
        else:
            padding = 'VALID'
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        input_channel = self.inputs.get_shape()[1]
        print("  [TL] Resnet_Convolution %s: kernel_size:%s strides:%s pad:%s activation:%s" %
              (self.name, str(kernel_size), str(strides), padding, act.__name__))

        with tf.variable_scope(name) as vs:
            if strides > 1:
                self.inputs = fixed_padding(self.inputs, kernel_size)
            W = tf.get_variable(name='W_conv2d', shape=[kernel_size, kernel_size, input_channel, filters],
                                initializer=W_init, **W_init_args)
            # print(tf.shape(W))
            if b_init:
                b = tf.get_variable(name='b_conv2d', shape=(filters), initializer=b_init, **b_init_args)
                self.outputs = act(tf.nn.conv2d(self.inputs, W, strides=[1, 1, strides, strides], padding=padding,
                                            use_cudnn_on_gpu=use_cudnn_on_gpu,
                                            data_format=data_format) + b)
            else:
                self.outputs = act(tf.nn.conv2d(self.inputs, W, strides=[1, 1, strides, strides], padding=padding,
                                            use_cudnn_on_gpu=use_cudnn_on_gpu,
                                            data_format=data_format))

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend([self.outputs])
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.extend([W])

def block_fn(inputs, filters, training, projection_shortcut,
                         strides, W_init, b_init):
    shortcut = inputs
    inputs = BatchNormLayer(inputs, act=tf.nn.relu, is_train=training, name='BN1')
    # The projection shortcut should come after the first batch norm and ReLU
    # since it performs a 1x1 convolution.
    if projection_shortcut is not None:
        with tf.variable_scope('shortcut'):
            shortcut = conv2d_fixed_padding(
            inputs,
            filters=filters * 4,
            kernel_size=1,
            strides=strides,
            act=tf.identity,
            W_init=W_init,
            b_init=b_init,
            name='projection_shortcut',
        )


    inputs = conv2d_fixed_padding(
        inputs,
        filters=filters,
        kernel_size=1,
        strides=1,
        act=tf.identity,
        W_init=W_init,
        b_init=b_init,
        name='conv2d_fixed_padding1',
    )
    inputs = BatchNormLayer(inputs, act=tf.nn.relu, is_train=training, name='BN2')
    inputs = conv2d_fixed_padding(
        inputs,
        filters=filters,
        kernel_size=3,
        strides=strides,
        act=tf.identity,
        W_init=W_init,
        b_init=b_init,
        name='conv2d_fixed_padding2',
    )
    inputs = BatchNormLayer(inputs, act=tf.nn.relu, is_train=training, name='BN3')
    inputs = conv2d_fixed_padding(
        inputs,
        filters=4 * filters,
        kernel_size=1,
        strides=1,
        act=tf.identity,
        W_init=W_init,
        b_init=b_init,
        name='conv2d_fixed_padding3',
    )
    inputs.outputs = inputs.outputs + shortcut.outputs
    inputs.outputs = tf.nn.relu(inputs.outputs)

    return inputs

def block_layer(inputs, filters, blocks, strides,
                training, W_init, b_init, name):

    with tf.variable_scope(name) as vs:

        # Only the first block per block_layer uses projection_shortcut and strides
        with tf.variable_scope('block{}'.format(1)):
            inputs = block_fn(inputs, filters, training, True, strides, W_init, b_init)

        for i in range(1, blocks):
            with tf.variable_scope('block{}'.format(i + 1)):
                inputs = block_fn(inputs, filters, training, None, 1, W_init, b_init)

    return inputs