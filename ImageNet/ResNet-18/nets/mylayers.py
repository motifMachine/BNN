from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig

from tensorlayer import _logging as logging
from tensorlayer.deprecation import deprecated_alias

def hard_sigmoid(x):
    return tf.clip_by_value((x + 1.) / 2., 0, 1)

def quantize(x):
    # ref: https://github.com/AngusG/tensorflow-xnor-bnn/blob/master/models/binary_net.py#L70
    #  https://github.com/itayhubara/BinaryNet.tf/blob/master/nnUtils.py
    with tf.get_default_graph().gradient_override_map({"Round": "Identity"}):
        #return ((tf.sign(x)+2)//2)*2-1
        return tf.round(hard_sigmoid(x))*2-1

class BinaryDenseLayer(Layer):
    """The :class:`BinaryDenseLayer` class is a binary fully connected layer, which weights are either -1 or 1 while inferencing.

    Note that, the bias vector would not be binarized.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_units : int
        The number of units of this layer.
    act : activation function
        The activation function of this layer, usually set to ``tf.act.sign`` or apply :class:`SignLayer` after :class:`BatchNormLayer`.
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inference. (TODO).
    W_init : initializer
        The initializer for the weight matrix.
    b_init : initializer or None
        The initializer for the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    name : a str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_units=100,
            act=tf.identity,
            use_gemm=False,
            W_init=tf.truncated_normal_initializer(stddev=0.1),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            name='binary_dense',
    ):
        super(BinaryDenseLayer, self).__init__(prev_layer=prev_layer, name=name)
        logging.info("BinaryDenseLayer  %s: %d %s" % (name, n_units, act.__name__))

        self.inputs = prev_layer.outputs

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}

        if self.inputs.get_shape().ndims != 2:
            raise Exception("The input dimension must be rank 2, please reshape or flatten it")

        if use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        n_in = int(self.inputs.get_shape()[-1])
        self.n_units = n_units

        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W', shape=(n_in, n_units), initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args
            )
            # W = tl.act.sign(W)    # dont update ...
            Wb = quantize(W)
            # W = tf.Variable(W)
            # print(W)
            if b_init is not None:
                try:
                    b = tf.get_variable(
                        name='b', shape=(n_units), initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args
                    )
                except Exception:  # If initializer is a constant, do not specify shape.
                    b = tf.get_variable(name='b', initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args)
                self.outputs = act(tf.matmul(self.inputs, Wb) + b)
                # self.outputs = act(xnor_gemm(self.inputs, W) + b) # TODO
            else:
                self.outputs = act(tf.matmul(self.inputs, Wb))
                # self.outputs = act(xnor_gemm(self.inputs, W)) # TODO

        self.all_layers.append(self.outputs)
        if b_init is not None:
            self.all_params.extend([W, b])
        else:
            self.all_params.append(W)


class BinaryConv2d(Layer):
    """
    The :class:`BinaryConv2d` class is a 2D binary CNN layer, which weights are either -1 or 1 while inference.

    Note that, the bias vector would not be binarized.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    n_filter : int
        The number of filters.
    filter_size : tuple of int
        The filter size (height, width).
    strides : tuple of int
        The sliding window strides of corresponding input dimensions.
        It must be in the same order as the ``shape`` parameter.
    act : activation function
        The activation function of this layer.
    padding : str
        The padding algorithm type: "SAME" or "VALID".
    use_gemm : boolean
        If True, use gemm instead of ``tf.matmul`` for inferencing. (TODO).
    W_init : initializer
        The initializer for the the weight matrix.
    b_init : initializer or None
        The initializer for the the bias vector. If None, skip biases.
    W_init_args : dictionary
        The arguments for the weight matrix initializer.
    b_init_args : dictionary
        The arguments for the bias vector initializer.
    use_cudnn_on_gpu : bool
        Default is False.
    data_format : str
        "NHWC" or "NCHW", default is "NHWC".
    name : str
        A unique layer name.

    Examples
    ---------
    >>> net = tl.layers.InputLayer(x, name='input')
    >>> net = tl.layers.BinaryConv2d(net, 32, (5, 5), (1, 1), padding='SAME', name='bcnn1')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool1')
    >>> net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train, name='bn1')
    ...
    >>> net = tl.layers.SignLayer(net)
    >>> net = tl.layers.BinaryConv2d(net, 64, (5, 5), (1, 1), padding='SAME', name='bcnn2')
    >>> net = tl.layers.MaxPool2d(net, (2, 2), (2, 2), padding='SAME', name='pool2')
    >>> net = tl.layers.BatchNormLayer(net, act=tl.act.htanh, is_train=is_train, name='bn2')

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            n_filter=32,
            filter_size=(3, 3),
            strides=(1, 1),
            act=tf.identity,
            padding='SAME',
            use_gemm=False,
            W_init=tf.truncated_normal_initializer(stddev=0.02),
            b_init=tf.constant_initializer(value=0.0),
            W_init_args=None,
            b_init_args=None,
            use_cudnn_on_gpu=None,
            data_format=None,
            # act=tf.identity,
            # shape=(5, 5, 1, 100),
            # strides=(1, 1, 1, 1),
            # padding='SAME',
            # W_init=tf.truncated_normal_initializer(stddev=0.02),
            # b_init=tf.constant_initializer(value=0.0),
            # W_init_args=None,
            # b_init_args=None,
            # use_cudnn_on_gpu=None,
            # data_format=None,
            name='binary_cnn2d',
    ):
        super(BinaryConv2d, self).__init__(prev_layer=prev_layer, name=name)
        logging.info(
            "BinaryConv2d %s: n_filter:%d filter_size:%s strides:%s pad:%s act:%s" %
            (name, n_filter, str(filter_size), str(strides), padding, act.__name__)
        )

        self.inputs = prev_layer.outputs

        if W_init_args is None:
            W_init_args = {}
        if b_init_args is None:
            b_init_args = {}
        if act is None:
            act = tf.identity
        if use_gemm:
            raise Exception("TODO. The current version use tf.matmul for inferencing.")

        if len(strides) != 2:
            raise ValueError("len(strides) should be 2.")
        try:
            pre_channel = int(prev_layer.outputs.get_shape()[-1])
        except Exception:  # if pre_channel is ?, it happens when using Spatial Transformer Net
            pre_channel = 1
            logging.info("[warnings] unknow input channels, set to 1")
        shape = (filter_size[0], filter_size[1], pre_channel, n_filter)
        strides = (1, strides[0], strides[1], 1)
        with tf.variable_scope(name):
            W = tf.get_variable(
                name='W_conv2d', shape=shape, initializer=W_init, dtype=LayersConfig.tf_dtype, **W_init_args
            )
            Wb = quantize(W)
            if b_init:
                b = tf.get_variable(
                    name='b_conv2d', shape=(shape[-1]), initializer=b_init, dtype=LayersConfig.tf_dtype, **b_init_args
                )
                self.outputs = act(
                    tf.nn.conv2d(
                        self.inputs, Wb, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
                        data_format=data_format
                    ) + b
                )
            else:
                self.outputs = act(
                    tf.nn.conv2d(
                        self.inputs, Wb, strides=strides, padding=padding, use_cudnn_on_gpu=use_cudnn_on_gpu,
                        data_format=data_format
                    )
                )

        self.all_layers.append(self.outputs)
        if b_init:
            self.all_params.extend([W, b])
        else:
            self.all_params.append(W)
            
class SignLayer(Layer):
    """The :class:`SignLayer` class is for quantizing the layer outputs to -1 or 1 while inferencing.

    Parameters
    ----------
    prev_layer : :class:`Layer`
        Previous layer.
    name : a str
        A unique layer name.

    """

    @deprecated_alias(layer='prev_layer', end_support_version=1.9)  # TODO remove this line for the 1.9 release
    def __init__(
            self,
            prev_layer,
            name='sign',
    ):
        super(SignLayer, self).__init__(prev_layer=prev_layer, name=name)

        self.inputs = prev_layer.outputs

        logging.info("SignLayer  %s" % (self.name))

        with tf.variable_scope(name):
            # self.outputs = tl.act.sign(self.inputs)
            self.outputs = quantize(self.inputs)

        self.all_layers.append(self.outputs)