# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#
import tensorflow as tf
import numpy as np
from utils import utils as helper
from core import initializers as initz
batch_norm_tf = tf.contrib.layers.batch_norm


def fully_connected(x, n_output, is_training, reuse, trainable=True, w_init=initz.he_normal(), b_init=0.0,
                    w_regularizer=tf.nn.l2_loss, name='fc', batch_norm=None, batch_norm_args=None, activation=None,
                    outputs_collections=None, use_bias=True):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) > 1, "Input Tensor shape must be > 1-D"
    if len(x.get_shape()) != 2:
        x = _flatten(x)

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name, reuse=reuse):
        shape = [n_input, n_output] if hasattr(w_init, '__call__') else None
        W = tf.get_variable(
            name='W',
            shape=shape,
            dtype=tf.float32,
            initializer=w_init,
            regularizer=w_regularizer,
            trainable=trainable
        )
        output = tf.matmul(x, W)

        if use_bias:
            b = tf.get_variable(
                name='b',
                shape=[n_output],
                dtype=tf.float32,
                initializer=tf.constant_initializer(b_init),
                trainable=trainable
            )

            output = tf.nn.bias_add(value=output, bias=b)

        if batch_norm is not None:
            if isinstance(batch_norm, bool):
                batch_norm = batch_norm_tf
            batch_norm_args = batch_norm_args or {}
            output = batch_norm(output, is_training=is_training, reuse=reuse, trainable=trainable, **batch_norm_args)

        if activation:
            output = activation(output, reuse=reuse, trainable=trainable)
    output.W = W
    if use_bias:
        output.b = b
    return output


def _flatten(x, name='flatten'):
    input_shape = helper.get_input_shape(x)
    assert len(input_shape) > 1, "Input Tensor shape must be > 1-D"
    with tf.name_scope(name):
        dims = int(np.prod(input_shape[1:]))
        flattened = tf.reshape(x, [-1, dims])
        return flattened
