import numpy as np
import tensorflow as tf
import os
import navirice_image_pb2
import cv2
import random
import sys

from navirice_generate_data import generate_bitmap_label
from navirice_helpers import navirice_image_to_np
from navirice_helpers import navirice_ir_to_np
from navirice_helpers import map_depth_and_rgb

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features):
    # unkown amount, higrt and width, channel
    input_layer = tf.reshape(features, [-1, 424, 512, 1])
    #scaled_layer = tf.image.resize_images(features, [53, 64])
    encoder1 = coder(max_pool_2x2(input_layer), [5,5,1,30], True)
    encoder2 = coder(max_pool_2x2(encoder1), [5,5,30,30], True)
    encoder3 = coder(max_pool_2x2(encoder2), [5,5,30,30], True)  
    decoder1 = coder(encoder3, [5,5,30,1], True)
    #last = tf.sigmoid(decoder1)
    last = decoder1

    h_final = tf.reshape(last, [-1, 53, 64]) 
    return h_final


def coder(input_layer, shape, do_relu):
    W_conv = weight_variable(shape)
    if do_relu:
        h_conv = tf.nn.leaky_relu(conv2d(input_layer, W_conv))
        #h_conv = tf.nn.lrn(h_conv, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
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
    y_ = tf.placeholder(tf.float32, [None, 424*scale_val, 512*scale_val])

    y_conv = cnn_model_fn(x)

    cost = tf.square(y_ - y_conv)
    #cost_mean = tf.reduce_sum(cost)
    #cost = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)
    #soft = tf.nn.softmax(logits=y_conv)
    #cost = tf.square(y_ - soft)
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
    
    s_train = False
    train_set_input = []
    train_set_expected =[]

    train_set_size = 10

    while(True):
        img_set, last_count = kc.navirice_get_image() 
        if(s_train):
            if(img_set != None and img_set.IR.width > 0 and img_set.Depth.width > 0):
                ir_image = navirice_ir_to_np(img_set.IR)
                depth_image = navirice_image_to_np(img_set.Depth)
                inverted_depth = np.ones(depth_image.shape)
                inverted_depth = inverted_depth - depth_image
                possible_bitmap = generate_bitmap_label(ir_image, depth_image)
                if possible_bitmap is not None: 
                    scaled_bitmap = cv2.resize(possible_bitmap,None,fx=scale_val, fy=scale_val, interpolation = cv2.INTER_CUBIC)
                    
                    if len(train_set_input) < train_set_size:
                        train_set_input.append(inverted_depth)
                        train_set_expected.append(scaled_bitmap)
                    else:
                        i = random.randint(0, train_set_size-1)
                        train_set_input[i] = inverted_depth
                        train_set_expected[i] = scaled_bitmap

                    train_step.run(session=sess, feed_dict={x: train_set_input, y_: train_set_expected})
                    cv2.imshow("idl", inverted_depth)
                    cv2.imshow("odl", scaled_bitmap)
        
        if(img_set != None and img_set.IR.width > 0 and img_set.Depth.width > 0):
            depth_image = navirice_image_to_np(img_set.Depth)
            ir_image = navirice_ir_to_np(img_set.IR)
            inverted_depth = np.ones(depth_image.shape)
            inverted_depth = inverted_depth - depth_image
             
            tests = [] 
            tests.append(inverted_depth)
            outs = sess.run(y_conv, feed_dict={x: tests})
            cv2.imshow("id",tests[0]) 
            cv2.imshow("od", outs[0])
        key = cv2.waitKey(10) & 0xFF
        print("key: ", key)

        if(key == 99):
            s_train = False

        if(key == 32):
            s_train = True

if __name__ == "__main__":
    main()

