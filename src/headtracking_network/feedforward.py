import numpy as np
import tensorflow as tf
import os
import navirice_image_pb2
import cv2
import random
import sys

from navirice_helpers import navirice_image_to_np
from navirice_helpers import navirice_ir_to_np
from navirice_helpers import map_depth_and_rgb

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features):
    # unkown amount, higrt and width, channel
    input_layer = tf.reshape(features, [-1, 424, 512, 1])

    mp0 = input_layer
    mp1 = max_pool_2x2(mp0)
    mp2 = max_pool_2x2(mp1)
    mp3 = max_pool_2x2(mp2)

    encoder1 = coder(mp1, [10,10,1,2], True)
    encoder2 = coder(mp2, [10,10,1,4], True)
    encoder3 = coder(mp3, [10,10,1,4], True)
    
    encoder4 = coder(encoder1, [10,10,2,4], True)
    encoder5 = coder(encoder2, [10,10,4,8], True)
    encoder6 = coder(encoder3, [10,10,4,8], True)


    W_fc1 = weight_variable([256*212*4, 1024])
    encoder4_last_flat = tf.reshape(encoder4, [-1, 256*212*4])
    h_fc1 = tf.matmul(encoder4_last_flat, W_fc1)
    
    W_fc2 = weight_variable([128*106*8, 1024])
    encoder5_last_flat = tf.reshape(encoder5, [-1, 128*106*8])
    h_fc2 = tf.matmul(encoder5_last_flat, W_fc2)
    

    W_fc3 = weight_variable([64*53*8, 1024])
    encoder6_last_flat = tf.reshape(encoder6, [-1, 64*53*8])
    h_fc3 = tf.matmul(encoder6_last_flat, W_fc3)
    
    merge_layer = tf.nn.sigmoid(h_fc3 + h_fc2 + h_fc1)

    W_fc2 = weight_variable([1024, 3])
    h_fc2 = tf.nn.sigmoid(tf.matmul(merge_layer, W_fc2))

    return h_fc2


def coder(input_layer, shape, do_relu):
    W_conv = weight_variable(shape)
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

def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def main():
    scale_val = 1.0/8.0
    x = tf.placeholder(tf.float32, [None, 424, 512, 1])
    y_ = tf.placeholder(tf.float32, [None, 3])
    y_conv = cnn_model_fn(x)

    #cost = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    cost = tf.square(y_ - y_conv)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init) 

    print("------------------OUT SHAPES-------------------")
    print(y_.get_shape())
    print(y_conv.get_shape()) 
    print("-----------------------------------------------")

    cnt = 0

    from navirice_get_image import KinectClient
    kc = KinectClient('127.0.0.1', 29000)
    kc.navirice_capture_settings(False, True, True)
    
    saver = tf.train.Saver()

    while(True):
        loc = input("Enter file destination to load: ")
        if(len(loc) > 0):
            try:
                saver.restore(sess, loc)
                break
            except ValueError:
                print("Error: no file with that destination")

    import time

    while(True):
        img_set, last_count = kc.navirice_get_image() 
       
        if(img_set != None and img_set.IR.width > 0 and img_set.Depth.width > 0):
            depth_image = navirice_image_to_np(img_set.Depth)
            ir_image = navirice_ir_to_np(img_set.IR)
            inverted_depth = np.ones(depth_image.shape)
            inverted_depth = inverted_depth - depth_image
             
            tests = [] 
            tests.append(inverted_depth)
            
            start = time.time()
            outs = sess.run(y_conv, feed_dict={x: tests})
            end = time.time()

            xf = outs[0][0]
            yf = outs[0][1]
            radiusf = outs[0][2]
            print("nnoutput x:", xf, "y: ", yf," r:", radiusf, " time elapsed(s): ", end-start)

if __name__ == "__main__":
    main()

