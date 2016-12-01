# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#

import tensorflow as tf
from core.layers import fully_connected as fc


class ActorModel(object):
    def __init__(self):
        self.end_points = {}

    def model_def(self, inputs, env, is_training=True, reuse=None, activation=tf.nn.relu, batchnorm=True, minval=-0.003, maxval=0.003, name=None):
        with tf.variable_scope('actor'):
            fc1 = fc(inputs, 400, is_training, reuse, activation=activation)
            self.end_points['fc1'] = fc1
            fc2 = fc(fc1, 300, is_training, reuse, activation=activation)
            self.end_points['fc2'] = fc2
            w_fc3_init = tf.random_uniform_initializer(minval=minval, maxval=maxval)
            out = fc(fc2, env.action_dim, is_training, reuse, activation=activation, w_init=w_fc3_init)
            self.end_points['out'] = out
            scaled_out = tf.mul(out, env.action_bound)
            self.end_points['scaled_out'] = scaled_out
            return self.end_points


class CriticModel(object):
    def __init__(self):
        self.end_points = {}

    def model_def(self, inputs, action_inputs, is_training=True, reuse=None, activation=tf.nn.relu, batchnorm=True, minval=-0.003, maxval=0.003, name=None):
        with tf.variable_scope('critic'):
            fc1 = fc(inputs, 400, is_training, reuse, activation=activation)
            self.end_points['fc1'] = fc1
            fc2_temp = fc(fc1, 300, is_training, reuse, activation=None)
            fc2_action = fc(action_inputs, 300, is_training, reuse, activation=None)
            fc2 = activation(tf.matmul(fc1, fc2_temp.W) + tf.matmul(action_inputs, fc2_action.W) + fc2_action.b, name='relu')
            self.end_points['fc2'] = fc2
            w_fc3_init = tf.random_uniform_initializer(minval=minval, maxval=maxval)
            out = fc(fc2, 1, is_training, reuse, activation=activation, w_init=w_fc3_init)
            self.end_points['out'] = out
            return self.end_points
