# -------------------------------------------------------------------#
# Released under the MIT license (https://opensource.org/licenses/MIT)
# Contact: mrinal.haloi11@gmail.com
# Copyright 2016, Mrinal Haloi
# -------------------------------------------------------------------#

import tensorflow as tf


class Model(object):

    def __init__(self, is_target_q=False):
        self.end_points = {}
        self.is_target_q = is_target_q

    def model_def(self, inputs, env, is_dueling=True, name=None):
        # scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        imageIn = tf.reshape(inputs, shape=[-1, 84, 84, 4])
        self.end_points['input'] = imageIn
        conv1 = tf.contrib.layers.convolution2d(inputs=imageIn, num_outputs=32, kernel_size=[
                                                8, 8], stride=[4, 4], padding='VALID', biases_initializer=None, scope=name + 'conv1')
        self.end_points['conv1'] = conv1
        conv2 = tf.contrib.layers.convolution2d(inputs=conv1, num_outputs=64, kernel_size=[
                                                4, 4], stride=[2, 2], padding='VALID', biases_initializer=None, scope=name + 'conv2')
        self.end_points['conv2'] = conv2
        conv3 = tf.contrib.layers.convolution2d(inputs=conv2, num_outputs=64, kernel_size=[
                                                3, 3], stride=[1, 1], padding='VALID', biases_initializer=None, scope=name + 'conv3')
        self.end_points['conv3'] = conv3
        conv4 = tf.contrib.layers.convolution2d(inputs=conv3, num_outputs=512, kernel_size=[
                                                7, 7], stride=[1, 1], padding='VALID', biases_initializer=None, scope=name + 'conv4')
        self.end_points['conv4'] = conv4
        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        if is_dueling:
            streamA, streamV = tf.split(conv4, 2, 3)
            streamA = tf.contrib.layers.flatten(
                streamA, scope=name + 'flatten1')
            self.end_points['flatten1'] = streamA
            streamV = tf.contrib.layers.flatten(
                streamV, scope=name + 'flatten2')
            self.end_points['flatten2'] = streamV
            v_fc1 = tf.contrib.layers.fully_connected(
                streamV, 512, scope=name + 'value_fc1')
            self.end_points['v_fc1'] = v_fc1
            a_fc1 = tf.contrib.layers.fully_connected(
                streamA, 512, scope=name + 'adv_fc1')
            self.end_points['a_fc1'] = a_fc1
            v_fc2 = tf.contrib.layers.fully_connected(
                v_fc1, 1, activation_fn=None, scope=name + 'value_fc2')
            self.end_points['v_fc2'] = v_fc2
            a_fc2 = tf.contrib.layers.fully_connected(
                a_fc1, env.action_size, activation_fn=None, scope=name + 'adv_fc2')
            self.end_points['a_fc2'] = a_fc2
            # Average Dueling
            q = v_fc2 + (a_fc2 - tf.reduce_mean(a_fc2,
                                                reduction_indices=1, keep_dims=True))
            self.end_points['q'] = q
        else:
            conv4_f = tf.contrib.layers.flatten(conv4, scope=name + 'flatten')
            fc1 = tf.contrib.layers.fully_connected(
                conv4_f, 512, scope=name + 'fc1')
            self.end_points['fc1'] = fc1
            q = tf.contrib.layers.fully_connected(
                fc1, env.action_size, scope=name + 'fc2')
            self.end_points['q'] = q

        if self.is_target_q:
            target_q_idx = tf.placeholder('int32', [None, None], 'output_idx')
            self.end_points['target_q_idx'] = target_q_idx
            target_q_with_idx = tf.gather_nd(q, target_q_idx)
            self.end_points['target_q_with_idx'] = target_q_with_idx

        pred_action = tf.argmax(q, dimension=1)
        self.end_points['pred_action'] = pred_action
        return self.end_points
