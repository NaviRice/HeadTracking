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
from models import cnn_model_fn1

tf.logging.set_verbosity(tf.logging.INFO)

def main():
    scale_val = 1.0/8.0
    x = tf.placeholder(tf.float32, [None, 424, 512, 1])
    y_ = tf.placeholder(tf.float32, [None, 3])
    y_conv, variables = cnn_model_fn1(x)

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
    
    saver = tf.train.Saver(variables)

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

