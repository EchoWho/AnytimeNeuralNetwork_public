#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: conv2d.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from .common import layer_register, VariableHolder
from ..utils.argtools import shape2d, shape4d
import numpy as np

__all__ = ['Conv2D', 'Deconv2D', 'AtrousConv2D', 'GroupedConv2D']


@layer_register()
def Conv2D(x, out_channel, kernel_shape,
           padding='SAME', stride=1,
           W_init=None, b_init=None, W_mask=None,
           nl=tf.identity, split=1, use_bias=True,
           data_format='NHWC'):
    """
    2D convolution on 4D inputs.

    Args:
        x (tf.Tensor): a 4D tensor.
            Must have known number of channels, but can have other unknown dimensions.
        out_channel (int): number of output channel.
        kernel_shape: (h, w) tuple or a int.
        stride: (h, w) tuple or a int.
        padding (str): 'valid' or 'same'. Case insensitive.
        split (int): Split channels as used in Alexnet. Defaults to 1 (no split).
        W_init: initializer for W. Defaults to `variance_scaling_initializer`.
        b_init: initializer for b. Defaults to zero.
        nl: a nonlinearity function.
        use_bias (bool): whether to use bias.

    Returns:
        tf.Tensor named ``output`` with attribute `variables`.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    in_shape = x.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Conv2D] Input cannot have unknown channel!"
    assert in_channel % split == 0
    assert out_channel % split == 0

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [in_channel / split, out_channel]
    stride = shape4d(stride, data_format=data_format)

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    if W_mask is not None:
        W = tf.multiply(W, W_mask)

    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)

    if split == 1:
        conv = tf.nn.conv2d(x, W, stride, padding, data_format=data_format)
    else:
        inputs = tf.split(x, split, channel_axis)
        kernels = tf.split(W, split, 3)
        outputs = [tf.nn.conv2d(i, k, stride, padding, data_format=data_format)
                   for i, k in zip(inputs, kernels)]
        conv = tf.concat(outputs, channel_axis)

    ret = nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')
    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret


class StaticDynamicShape(object):
    def __init__(self, static, dynamic):
        self.static = static
        self.dynamic = dynamic

    def apply(self, f):
        try:
            st = f(self.static)
            return StaticDynamicShape(st, st)
        except:
            return StaticDynamicShape(None, f(self.dynamic))


def get_deconv_filter(f_shape):
    width = f_shape[0]
    heigh = f_shape[1]
    f = np.ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - np.abs(x / f - c)) * (1 - np.abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    #init = tf.constant_initializer(value=weights,
    #                               dtype=tf.float32)
    #return tf.get_variable(name="up_filter", initializer=init,
    #                       shape=weights.shape)
    return tf.constant(weights, dtype=tf.float32)


@layer_register()
def Deconv2D(x, out_shape, kernel_shape,
             stride, padding='SAME',
             dyn_hw=None, 
             W_init=None, b_init=None,
             nl=tf.identity, use_bias=True,
             data_format='NHWC'):
    """
    2D deconvolution on 4D inputs.

    Args:
        x (tf.Tensor): a tensor of shape NHWC.
            Must have known number of channels, but can have other unknown dimensions.
        out_shape: (h, w, channel) tuple, or just a integer channel,
            then (h, w) will be calculated by input_shape * stride
        kernel_shape: (h, w) tuple or a int.
        stride: (h, w) tuple or a int.
        padding (str): 'valid' or 'same'. Case insensitive.
        dyn_hw: (h, w) both are dynamic shape for the output. e.g., dyn_h = tf.shape(OutTarget)[H], 
        W_init: initializer for W. Defaults to `variance_scaling_initializer`.
        b_init: initializer for b. Defaults to zero.
        nl: a nonlinearity function.
        use_bias (bool): whether to use bias.

    Returns:
        tf.Tensor: a NHWC tensor named ``output`` with attribute `variables`.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """
    in_shape = x.get_shape().as_list()
    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]
    assert in_channel is not None, "[Deconv2D] Input cannot have unknown channel!"
    kernel_shape = shape2d(kernel_shape)
    stride2d = shape2d(stride)
    stride4d = shape4d(stride, data_format=data_format)
    padding = padding.upper()
    in_shape_dyn = tf.shape(x)

    if isinstance(out_shape, int) and dyn_hw is None:
        out_channel = out_shape
        if data_format == 'NHWC':
            shp3_0 = StaticDynamicShape(in_shape[1], in_shape_dyn[1]).apply(lambda x: stride2d[0] * x)
            shp3_1 = StaticDynamicShape(in_shape[2], in_shape_dyn[2]).apply(lambda x: stride2d[1] * x)
            shp3_dyn = [shp3_0.dynamic, shp3_1.dynamic, out_channel]
            shp3_static = [shp3_0.static, shp3_1.static, out_channel]
        else:
            shp3_0 = StaticDynamicShape(in_shape[2], in_shape_dyn[2]).apply(lambda x: stride2d[0] * x)
            shp3_1 = StaticDynamicShape(in_shape[3], in_shape_dyn[3]).apply(lambda x: stride2d[1] * x)
            shp3_dyn = [out_channel, shp3_0.dynamic, shp3_1.dynamic]
            shp3_static = [out_channel, shp3_0.static, shp3_1.static]
    elif isinstance(out_shape, int) and isinstance(dyn_hw, list):
        out_channel = out_shape
        if data_format == 'NHWC':
            shp3_dyn = dyn_hw + [out_channel]
            shp3_static = [None, None, out_channel]
        else:
            shp3_dyn = [out_channel] + dyn_hw
            shp3_static = [out_channel, None, None]
    else:
        for k in out_shape:
            if not isinstance(k, int):
                raise ValueError("[Deconv2D] out_shape {} is invalid!".format(k))
        out_channel = out_shape[channel_axis - 1]   # out_shape doesn't have batch
        shp3_static = shp3_dyn = out_shape
    filter_shape = kernel_shape + [out_channel, in_channel]

    if b_init is None:
        b_init = tf.constant_initializer()
    W = tf.get_variable('W', filter_shape, initializer=W_init)
    #W_bias = get_deconv_filter(filter_shape)
    W_sum = W #+ W_bias
    if use_bias:
        b = tf.get_variable('b', [out_channel], initializer=b_init)

    out_shape_dyn = tf.stack([tf.shape(x)[0]] + shp3_dyn)
    conv = tf.nn.conv2d_transpose(
        x, W_sum, out_shape_dyn, stride4d, padding=padding, data_format=data_format)
    conv.set_shape(tf.TensorShape([None] + shp3_static))
    ret = nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')

    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret


@layer_register()
def AtrousConv2D(x, ch_out, kernel_shape, dilation_rate,
                 padding='SAME',
                 W_init=None, b_init=None,
                 nl=tf.identity, use_bias=True,
                 data_format='NHWC'):

    in_shape = x.get_shape().as_list()
    if data_format == 'NHWC':
        ch_axis = 3
        h_axis = 1
        w_axis = 2
    else:
        ch_axis = 1
        h_axis = 2
        w_axis = 3

    ch_in = in_shape[ch_axis] 
    h_in = in_shape[h_axis]
    w_in = in_shape[w_axis]
    kernel_shape = shape2d(kernel_shape)
    filter_shape = kernel_shape + [ch_in, ch_out]
    dilation_rate = shape2d(dilation_rate)
    h_out = None if h_in is None else h_in * dilation_rate[0]
    w_out = None if w_in is None else w_in * dilation_rate[1]
    if data_format == 'NHWC':
        out_shape = [None, h_out, w_out, ch_out]
    else:
        out_shape = [None, ch_out, h_out, w_out]
    padding = padding.upper()

    if W_init is None:
        W_init = tf.contrib.layers.xavier_initializer_conv2d()
    W = tf.get_variable('W', filter_shape, initializer=W_init)
    if use_bias:
        if b_init is None:
            b_init = tf.constant_initializer()
        b = tf.get_variable('b', [ch_out], initializer=b_init)

    conv = tf.nn.convolution(x, W, padding, strides=None, 
        dilation_rate=dilation_rate, data_format=data_format)
    
    conv.set_shape(tf.TensorShape(out_shape))
    ret = nl(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')
    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret


@layer_register()
def GroupedConv2D(x, num_paths, path_ch_out, kernel_shape,
        sum_paths=False, padding='SAME', stride=1, 
        W_init=None, b_init=None, nl=tf.identity,
        use_bias=False, data_format='NHWC'):
    """
    Grouped conv 2d for ResNeXt. Uses depthwise conv 2d and reshape and sum.
   
    Args:
        x : 4D tensor of data_format
        num_paths : number of groups
        path_ch_out : number of ch_out per group
        kernel_shape : (h,w) tuple or an int
        sum_paths : whether the groups are summed together (if True) 
            or concatenated (if False (default))
        padding, W_init, b_init, nl, use_bias, data_format : see Conv2D

    Returns:
        tf.Tensor named ``output`` with attribute `variables`.

    Variable Names:

    * ``W``: weights
    * ``b``: bias
    """

    in_shape = x.get_shape().as_list()
    ch_dim = 3 if data_format == 'NHWC' else 1
    ch_in = in_shape[ch_dim]
    assert ch_in % num_paths == 0, "Grouped conv requires n_groups to divide ch_in" 
    ch_in_per_path = ch_in // num_paths
    ch_out = path_ch_out if sum_paths else num_paths * path_ch_out

    kernel_shape = shape2d(kernel_shape)
    padding = padding.upper()
    filter_shape = kernel_shape + [ch_in, path_ch_out]
    stride = shape4d(stride, data_format=data_format)

    if W_init is None:
        W_init = tf.contrib.layers.variance_scaling_initializer()
    if b_init is None:
        b_init = tf.constant_initializer()

    W = tf.get_variable('W', filter_shape, initializer=W_init)
    if use_bias:
        b = tf.get_variable('b', [ch_out], initializer=b_init)

    x = tf.nn.depthwise_conv2d(x, W, stride, padding, rate=None, data_format=data_format)
    out_shape = x.get_shape().as_list()

    # First reshape to expose the dimension by input channels 
    shape_depthwise = [num_paths, ch_in_per_path, path_ch_out]
    if data_format == 'NHWC':
        x = tf.reshape(x, [-1, out_shape[1], out_shape[2]] + shape_depthwise)
    else:
        x = tf.reshape(x, [-1] + shape_depthwise + [out_shape[2], out_shape[3]])

    # Then reduce sum to remove the input channel leaving output dim and (path dim)
    if sum_paths:
        sum_axis = [ch_dim, ch_dim + 1]
    else:
        sum_axis = ch_dim + 1
    x = tf.reduce_sum(x, sum_axis) 

    # reshape to output shape if path dim did not collapse
    if not sum_paths:
        if data_format == 'NHWC':
            x = tf.reshape(x, [-1, out_shape[1], out_shape[2], ch_out])
        else:
            x = tf.reshape(x, [-1, ch_out, out_shape[2], out_shape[3]])

    ret = nl(tf.nn.bias_add(x, b, data_format=data_format) if use_bias else x, name='output')
    ret.variables = VariableHolder(W=W)
    if use_bias:
        ret.variables.b = b
    return ret
