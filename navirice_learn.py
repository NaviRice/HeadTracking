import numpy as np
import tensorflow as tf
import os
import navirice_image_pb2
import cv2
import random
import sys

from navirice_generate_data import generate_bitmap_label
from navirice_helpers import navirice_image_to_np
from navirice_helpers import map_depth_and_rgb

tf.logging.set_verbosity(tf.logging.INFO)


def load_data_file_list(path):
    dir_file_list = os.listdir(path)
    data_file_list = []
    for item in dir_file_list:
        if item.endswith('.img_set'):
            data_file_list.append(path + "/" + item)
    return data_file_list


# Sample output
'''
======== FILE STATS ========
Path:    ./DATA/default_3718.img_set
RGB:     1920 x 1080
DEPTH:   512 x 424
IR:  512 x 424
======== END ========
'''
def print_image_stats(path):    
    data_file_list = load_data_file_list(path)
    pick = random.choice(data_file_list) 
    f = open(pick, "rb")
    data = f.read()
    img_set = navirice_image_pb2.ProtoImageSet()
    img_set.ParseFromString(data)
    print("======== FILE STATS ========")
    print("Path:\t", pick)
    print("RGB:\t", img_set.RGB.width, "x", img_set.RGB.height)
    print("DEPTH:\t", img_set.Depth.width, "x", img_set.Depth.height)
    print("IR:\t", img_set.IR.width, "x", img_set.IR.height)
    print("======== END ========")

def load_all(data_list, scale_val):
    real = []
    expected = []
    for i in range(len(data_list)):
        with open(data_list[i], 'rb') as ci:
            data=ci.read()
            img_set = navirice_image_pb2.ProtoImageSet()
            img_set.ParseFromString(data)
            del data
            if img_set.RGB is not None and img_set.Depth is not None:
                rgb_image = navirice_image_to_np(img_set.RGB)
                depth_image = navirice_image_to_np(img_set.Depth)
                (rgb_image, depth_image) = map_depth_and_rgb(rgb_image, depth_image)
                possible_bitmap = generate_bitmap_label(rgb_image, depth_image)
                if possible_bitmap is not None:
                    print(depth_image.shape)
                    real.append(depth_image)
                    scaled_bitmap = cv2.resize(possible_bitmap,None,fx=scale_val, fy=scale_val, interpolation = cv2.INTER_CUBIC)
                    expected.append(scaled_bitmap)
            del img_set
    return (real, expected)


def generate_batch(count, real_list, expected_list):
    i = 0
    real = []
    expected = []
    while i < count:
        ri = random.randint(0, len(real_list)-1)
        real.append(real_list[ri])
        expected.append(expected_list[ri])
        i += 1
    return (real, expected)


def cnn_model_fn(features):
    input_layer = tf.reshape(features, [-1, 364, 512, 1])
    
    encoder1 = coder(input_layer, [5,5,1,32], True)
    pool1 = max_pool_2x2(encoder1)
    encoder2 = coder(pool1, [5,5,32,64], True)
    pool2 = max_pool_2x2(encoder2)
    encoder3 = coder(pool2, [5,5,64,16], True)  
    decoder1 = coder(encoder3, [5,5,16,1], False) 

    h_final = tf.reshape(decoder1, [-1, 91, 128]) 
    return h_final

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
    data_list = load_data_file_list("./DATA")
    (reals, expecteds) = load_all(data_list, 1.0/4)
    print("real's shape: " + str(reals[0].shape))
    print("expected's shape: " + str(expecteds[0].shape))

    x = tf.placeholder(tf.float32, [None, 364, 512, 1])
    y_ = tf.placeholder(tf.float32, [None, 91, 128])

    y_conv = cnn_model_fn(x)

    #cost = tf.square(y_ - y_conv)
    #cost_mean = tf.reduce_sum(cost)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init) 

    print("------------------OUT SHAPES-------------------")
    print(y_.get_shape())
    print(y_conv.get_shape()) 
    print("-----------------------------------------------")

    cnt = 0
    while(True):
        cnt += 1
        print("STEP COUNT: ", cnt)
        for i in range(100):
            (reals_i, expecteds_i) = generate_batch(10, reals, expecteds)
            print("-", end='')
            sys.stdout.flush()
            train_step.run(session=sess, feed_dict={x: reals_i, y_: expecteds_i})
        print("|")
            # see the first image
       
        for i in range(10):
            (reals_i, expecteds_i) = generate_batch(10, reals, expecteds)
            outs = sess.run(y_conv, feed_dict={x: reals_i})
            for i in range(len(reals_i)):
                cv2.imshow("input", reals_i[i])
                cv2.imshow("expected", expecteds_i[i])
                cv2.imshow("output", outs[i])
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break

if __name__ == "__main__":
    main()

