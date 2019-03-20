from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import sys
import types as python_types
import warnings

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.util.tf_export import tf_export

def _l2normalizer(v, epsilon=1e-12):
    return v / (K.sum(v ** 2) ** 0.5 + epsilon)


def power_iteration(W, u, rounds=1):
    '''
    Accroding the paper, we only need to do power iteration one time.
    '''
    _u = u

    for i in range(rounds):
        _v = _l2normalizer(K.dot(_u, W))
        _u = _l2normalizer(K.dot(_v, K.transpose(W)))

    W_sn = K.sum(K.dot(_u, W) * _v)
    return W_sn, _u, _v

def compute_spectral_normal(self, out_channel_axis=-1, training=True):
    # Spectrally Normalized Weight
    if self.spectral_normalization:
        # Get kernel tensor shape [batch, units]
        W_shape = self.kernel.shape.as_list()

        # Flatten the Tensor
        W_mat = K.reshape(self.kernel, [W_shape[out_channel_axis], -1])  # [out_channels, N]

        W_sn, u, _ = power_iteration(W_mat, self.u)

        if training:
            # Update estimated 1st singular vector
            self.u.assign(u)

        return self.kernel / W_sn
    else:
        return self.kernel    

def init(self, units, spectral_normalization):
    if spectral_normalization:
            self.u = K.random_normal_variable([1, units], 0, 1, name="sn_estimate")  # [1, out_channels]
            self.spectral_normalization = spectral_normalization

class Dense(layers.Dense):
    def __init__(self, units,
                 spectral_normalization=True, **kwargs):
        init(self, units, spectral_normalization)
        super(Dense, self).__init__(units=units, **kwargs)

    def call(self, inputs, training=False):
        inputs = ops.convert_to_tensor(inputs)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, compute_spectral_normal(self, training=training), [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, compute_spectral_normal(self, training=training))
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

class Conv2D(layers.Conv2D):
    def __init__(self, filters, kernel_size, spectral_normalization=True, **kwargs):
        init(self, filters, spectral_normalization)
        super(Conv2D, self).__init__(filters, kernel_size, **kwargs)


    def call(self, inputs, training=False):
        outputs = self._convolution_op(inputs, compute_spectral_normal(self, training=training))

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                if self.rank == 2:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
                if self.rank == 3:
                    # As of Mar 2017, direct addition is significantly slower than
                    # bias_add when computing gradients. To use bias_add, we collapse Z
                    # and Y into a single dimension to obtain a 4D input tensor.
                    outputs_shape = outputs.shape.as_list()
                    if outputs_shape[0] is None:
                        outputs_shape[0] = -1
                    outputs_4d = array_ops.reshape(outputs,
                                                    [outputs_shape[0], outputs_shape[1],
                                                    outputs_shape[2] * outputs_shape[3],
                                                    outputs_shape[4]])
                    outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class Conv2DTranspose(layers.Conv2DTranspose):
    def __init__(self, filters, kernel_size, spectral_normalization=True, **kwargs):
        init(self, filters, spectral_normalization)
        super(Conv2DTranspose, self).__init__(filters, kernel_size, **kwargs)
    
    def call(self, inputs, training=False):
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(height,
                                                        kernel_h,
                                                        padding=self.padding,
                                                        output_padding=out_pad_h,
                                                        stride=stride_h,
                                                        dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width,
                                                        kernel_w,
                                                        padding=self.padding,
                                                        output_padding=out_pad_w,
                                                        stride=stride_w,
                                                        dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = array_ops.stack(output_shape)
        outputs = K.conv2d_transpose(
                inputs,
                compute_spectral_normal(self, out_channel_axis=-2, training=training),
                output_shape_tensor,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = nn.bias_add(
                    outputs,
                    self.bias,
                    data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs



