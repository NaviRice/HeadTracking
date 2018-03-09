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
    
    s_train = False
    r_train = False
    train_set_input = []
    train_set_expected =[]

    train_set_size = 100000

    saver = tf.train.Saver(variables)

    while(True):
        img_set, last_count = kc.navirice_get_image() 
        if(s_train):
            s_train = False
            if(img_set != None and img_set.IR.width > 0 and img_set.Depth.width > 0):
                ir_image = navirice_ir_to_np(img_set.IR)
                #depth_image = navirice_image_to_np(img_set.Depth)
                #inverted_depth = np.ones(depth_image.shape)
                #inverted_depth = inverted_depth - depth_image
                cv_result = get_head_from_img(ir_image)
                if cv_result is not None: 
                    arr = [cv_result[0], cv_result[1], cv_result[2]]
                    if len(train_set_input) < train_set_size:
                        train_set_input.append(ir_image)
                        train_set_expected.append(arr)
                    else:
                        if(random.randint(0, 10000) > -1):
                            i = random.randint(0, train_set_size-1)
                            train_set_input[i] = ir_image
                            train_set_expected[i] = arr
                
                    #train_step.run(session=sess, feed_dict={x: train_set_input, y_: train_set_expected})
                    dp = ir_image.copy()
                    cv2.circle(dp, (int(cv_result[0]*512), int(cv_result[1]*424)), int(cv_result[2]*400), (255, 0, 0), thickness=3, lineType=8, shift=0)
                    cv2.imshow("idl", dp)
                    print("db count: ", len(train_set_input))
        
        if(img_set != None and img_set.IR.width > 0 and img_set.Depth.width > 0):
            #depth_image = navirice_image_to_np(img_set.Depth)
            ir_image = navirice_ir_to_np(img_set.IR)
            #inverted_depth = np.ones(depth_image.shape)
            #inverted_depth = inverted_depth - depth_image
             
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

        if(r_train):
            tsi=[]
            tse=[]
            for i in range(100):
                random_index = random.randint(0, len(train_set_input)-1)
                tsi.append(train_set_input[random_index])
                tse.append(train_set_expected[random_index])
            print("TRAINING")
            train_step.run(session=sess, feed_dict={x: tsi, y_: tse})

        key = cv2.waitKey(10) & 0xFF
        #print("key: ", key)

        # train
        if(key == ord('t')):
            r_train = True

        # rest
        if(key == ord('r')):
            r_train = False

        # (space) capture
        if(key == 32):
            s_train = True

        # save model
        if(key == ord('s')):
            loc = input("Enter file destination to save: ")
            if(len(loc) > 0):
                try:
                    saver.save(sess, loc)
                except ValueError:
                    print("Error: Did not enter a path..")
        
        # load model
        if(key == ord('l')):
            loc = input("Enter file destination to load: ")
            if(len(loc) > 0):
                try:
                    saver.restore(sess, loc)
                except ValueError:
                    print("Error: no file with that destination")


if __name__ == "__main__":
    main()

