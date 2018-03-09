import numpy as np
import tensorflow as tf
import os
import navirice_image_pb2
import random
import sys

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn1(features):
    # unkown amount, higrt and width, channel
    input_layer = tf.reshape(features, [-1, 424, 512, 1])

    variables = []

    mp0 = input_layer
    mp1 = max_pool_2x2(mp0)
    mp2 = max_pool_2x2(mp1)
    mp3 = max_pool_2x2(mp2)

    #encoder1 = coder(mp1, [10,10,1,2], True, variables)
    encoder2 = coder(mp2, [10,10,1,4], True, variables)
    encoder3 = coder(mp3, [10,10,1,4], True, variables)
    
    #encoder4 = coder(encoder1, [10,10,2,4], True, variables)
    encoder5 = coder(encoder2, [10,10,4,8], True, variables)
    encoder6 = coder(encoder3, [10,10,4,8], True, variables)


    #W_fc1 = weight_variable([256*212*4, 1024], variables)
    #encoder4_last_flat = tf.reshape(encoder4, [-1, 256*212*4])
    #h_fc1 = tf.matmul(encoder4_last_flat, W_fc1)
    
    W_fc2 = weight_variable([128*106*8, 1024], variables)
    encoder5_last_flat = tf.reshape(encoder5, [-1, 128*106*8])
    h_fc2 = tf.matmul(encoder5_last_flat, W_fc2)
    

    W_fc3 = weight_variable([64*53*8, 1024], variables)
    encoder6_last_flat = tf.reshape(encoder6, [-1, 64*53*8])
    h_fc3 = tf.matmul(encoder6_last_flat, W_fc3)
    
    merge_layer = tf.nn.sigmoid(h_fc3 + h_fc2) #+ h_fc1)

    W_fc2 = weight_variable([1024, 3], variables)
    h_fc2 = tf.nn.sigmoid(tf.matmul(merge_layer, W_fc2))

    return h_fc2, variables


def coder(input_layer, shape, do_relu, variables):
    W_conv = weight_variable(shape, variables)
    if do_relu:
        h_conv = tf.nn.leaky_relu(conv2d(input_layer, W_conv))
        return h_conv
    else:
        h_conv = conv2d(input_layer, W_conv)
        return h_conv

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape, variables):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    v = tf.Variable(initial)
    variables.append(v)
    return v

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
