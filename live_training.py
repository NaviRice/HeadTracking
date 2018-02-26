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
    
    s_train = False
    r_train = False
    train_set_input = []
    train_set_expected =[]

    train_set_size = 100000

    saver = tf.train.Saver()

    while(True):
        img_set, last_count = kc.navirice_get_image() 
        if(s_train):
            s_train = False
            if(img_set != None and img_set.IR.width > 0 and img_set.Depth.width > 0):
                ir_image = navirice_ir_to_np(img_set.IR)
                depth_image = navirice_image_to_np(img_set.Depth)
                inverted_depth = np.ones(depth_image.shape)
                inverted_depth = inverted_depth - depth_image
                cv_result = get_head_from_img(ir_image)
                if cv_result is not None: 
                    arr = [cv_result[0], cv_result[1], cv_result[2]]
                    if len(train_set_input) < train_set_size:
                        train_set_input.append(inverted_depth)
                        train_set_expected.append(arr)
                    else:
                        if(random.randint(0, 10000) > -1):
                            i = random.randint(0, train_set_size-1)
                            train_set_input[i] = inverted_depth
                            train_set_expected[i] = arr
                
                    #train_step.run(session=sess, feed_dict={x: train_set_input, y_: train_set_expected})
                    dp = inverted_depth.copy()
                    cv2.circle(dp, (int(cv_result[0]*512), int(cv_result[1]*424)), int(cv_result[2]*400), (255, 0, 0), thickness=3, lineType=8, shift=0)
                    cv2.imshow("idl", dp)
                    print("db count: ", len(train_set_input))
        
        if(img_set != None and img_set.IR.width > 0 and img_set.Depth.width > 0):
            depth_image = navirice_image_to_np(img_set.Depth)
            ir_image = navirice_ir_to_np(img_set.IR)
            inverted_depth = np.ones(depth_image.shape)
            inverted_depth = inverted_depth - depth_image
             
            tests = [] 
            tests.append(inverted_depth)
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

