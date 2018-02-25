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
from navirice_head_detect import get_head_from_img

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features):
    # unkown amount, higrt and width, channel
    input_layer = tf.reshape(features, [-1, 424, 512, 1])
    encoder1 = coder(max_pool_2x2(input_layer), [10,10,1,12], True)
    encoder2 = coder(max_pool_2x2(encoder1), [7,7,12,24], True)
    encoder3 = coder((encoder2), [5,5,24,64], True)
    #encoder4 = coder((encoder3), [7,7,64,64], True)
    #encoder5 = coder((encoder4), [7,7,64,64], True)
    #encoder6 = coder((encoder5), [7,7,64,64], True)
    
    encoder_last = coder(max_pool_2x2(encoder3), [7,7,64,64], True)  

    W_fc1 = weight_variable([64*53*64, 1024])
    encoder_last_flat = tf.reshape(encoder_last, [-1, 64*53*64])
    h_fc1 = tf.nn.sigmoid(tf.matmul(encoder_last_flat, W_fc1))
    
    W_fc2 = weight_variable([1024, 3])
    h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2))

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
    
    s_train = False
    train_set_input = []
    train_set_expected =[]

    train_set_size = 100

    while(True):
        img_set, last_count = kc.navirice_get_image() 
        if(s_train):
            if(img_set != None and img_set.IR.width > 0 and img_set.Depth.width > 0):
                ir_image = navirice_ir_to_np(img_set.IR)
                depth_image = navirice_image_to_np(img_set.Depth)
                inverted_depth = np.ones(depth_image.shape)
                inverted_depth = inverted_depth - depth_image
                cv_result = get_head_from_img(ir_image)
                if cv_result is not None: 
                    arr = [cv_result[0], cv_result[1], cv_result[2]]
                    if len(train_set_input) < train_set_size:
                        train_set_input.append(ir_image)
                        train_set_expected.append(arr)
                    else:
                        if(random.randint(0, 10000) > 5000):
                            i = random.randint(0, train_set_size-1)
                            train_set_input[i] = ir_image
                            train_set_expected[i] = arr
                
                    train_step.run(session=sess, feed_dict={x: train_set_input, y_: train_set_expected})
                    dp = ir_image.copy()
                    cv2.circle(dp, (int(cv_result[0]*512), int(cv_result[1]*424)), int(cv_result[2]*400), (255, 0, 0), thickness=3, lineType=8, shift=0)
                    cv2.imshow("idl", dp)
        
        if(img_set != None and img_set.IR.width > 0 and img_set.Depth.width > 0):
            depth_image = navirice_image_to_np(img_set.Depth)
            ir_image = navirice_ir_to_np(img_set.IR)
            inverted_depth = np.ones(depth_image.shape)
            inverted_depth = inverted_depth - depth_image
             
            tests = [] 
            tests.append(ir_image)
            outs = sess.run(y_conv, feed_dict={x: tests})
            xf = outs[0][0]
            yf = outs[0][1]
            radiusf = outs[0][2]
            print("nnoutput x:", xf, "y: ", yf," r:", radiusf)
            if radiusf < 0:
                radiusf = 0

            cv2.circle(tests[0], (int(xf*512), int(yf*424)), int(radiusf*400), (255, 0, 0), thickness=3, lineType=8, shift=0)
            cv2.imshow("id",tests[0]) 

        key = cv2.waitKey(10) & 0xFF
        #print("key: ", key)

        if(key == 99):
            s_train = False

        if(key == 32):
            s_train = True

if __name__ == "__main__":
    main()

