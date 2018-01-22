# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def layer_width(layer):
    return layer.get_shape().as_list()[1]

def ndarray_length(layer):
    return layer.shape[0]

weights = []

def make_linear_layer(inputs, size, regularize = True):

    W = tf.Variable(tf.random_normal(
        [layer_width(inputs), size], 
        stddev = 1. / np.sqrt(layer_width(inputs))
    ))

    if regularize:
        weights.append(W)

    b = tf.Variable(tf.zeros([size]))
    return tf.matmul(inputs, W) + b

def make_layer(inputs, size):
    return tf.nn.sigmoid(make_linear_layer(inputs, size))

def main(_):
    L2_REGULARIZATION_FACTOR = 0.0001
    MINIBATCH_SIZE = 100
    MAX_STRIKES = 20
    INITIAL_LEARNING_RATE = 0.8
    LEARNING_RATE_BACKDOWN = 0.7
    INITIAL_LEARNING_MOMENTUM = 0.2
    LEARNING_MOMENTUM_BACKDOWN = 0.5

    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    #TODO ACTUALLY CONSIDER IMAGES AS 2D

    #augment data
    print('Generating more rotated images')
    num_images = ndarray_length(mnist.train.images)
    print(num_images)
    mnist.train.images = tf.contrib.image.rotate(mnist.train.images, tf.zeros(num_images))

    # TODO TRY ROTATING
    # TODO TRY INVERTING COLORS

    # Create the model
    inputs = tf.placeholder(tf.float32, [None, 784])

    lastLayer = inputs
    lastLayer = make_layer(lastLayer, 300)
    # lastLayer = make_layer(lastLayer, 100)
    lastLayer = make_linear_layer(lastLayer, 10, False)

    # Define loss and optimizer
    labels = tf.placeholder(tf.float32, [None, 10])
    
    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(labels * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=lastLayer))

    #add regularization
    if L2_REGULARIZATION_FACTOR > 0:
        for w in weights:
            loss = loss + L2_REGULARIZATION_FACTOR * tf.nn.l2_loss(w)

    learning_rate = tf.placeholder(tf.float32, shape=[])
    learning_momentum = tf.placeholder(tf.float32, shape=[])
    train_step = tf.train.MomentumOptimizer(learning_rate, learning_momentum).minimize(loss)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    # Train
    last_accuracy = 0
    strikes = 0
    epoch = 0
    learning_rate_f = INITIAL_LEARNING_RATE
    learning_momentum_f = INITIAL_LEARNING_MOMENTUM
    while True:
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(MINIBATCH_SIZE)
            sess.run(train_step, feed_dict={
                     inputs: batch_xs, 
                     labels: batch_ys,
                     learning_rate: learning_rate_f,
                     learning_momentum: learning_momentum_f})

        # Test trained model
        correct_prediction = tf.equal(
            tf.argmax(lastLayer, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={inputs: mnist.test.images,
                                               labels: mnist.test.labels})

        if result < last_accuracy:
            strikes += 1

            #decrease the learning rate every time something gets wrong
            #in theory, this lets us find finer optimizations
            learning_rate_f *= LEARNING_RATE_BACKDOWN
            learning_momentum_f *= LEARNING_MOMENTUM_BACKDOWN

            if strikes > MAX_STRIKES:
                return
                # TODO instead of failing, save the last good state
                # then generate n "offspring" with random variation,
                # and continue from there
                # pick the offspring with the best value
                # if the best is still worse end
        else:
            strikes = 0
            learning_rate_f = INITIAL_LEARNING_RATE
            learning_momentum_f = INITIAL_LEARNING_MOMENTUM

            print(str(epoch) + ":\taccuracy "  + str(result * 100) + "%")
            last_accuracy = result

        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
