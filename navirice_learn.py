import numpy as np
import tensorflow as tf
import os
import navirice_image_pb2
import cv2
import random

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

def generate_batch(count, data_list, scale_val):
    i = 0
    real = []
    expected = []
    while i < count:
        ri = random.randint(0, len(data_list)-1)
        with open(data_list[ri], 'rb') as ci:
            data=ci.read()
            img_set = navirice_image_pb2.ProtoImageSet()
            img_set.ParseFromString(data)
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
                    i += 1
    return (real, expected)


def cnn_model_fn(features, labels):
    input_layer = tf.reshape(features, [-1, 364, 512, 1])

    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(input_layer, W_conv1))

    h_pool2 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool2, W_conv2))

    h_pool3 = max_pool_2x2(h_conv2)
    # img down to 91 by 128 by 64
    print("h_pool3: ", h_pool3.get_shape())

    W_conv3 = weight_variable([5,5,64,32])
    b_conv3 = bias_variable([32])
    h_conv3 = tf.nn.relu(conv2d(h_pool3, W_conv3))

    W_conv4 = weight_variable([5,5,32,1])
    b_conv4 = bias_variable([1])
    h_conv4 = conv2d(h_conv3, W_conv4)
 
    h_final = tf.reshape(h_conv4, [-1, 91, 128]) 
    return h_final




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
    (reals, expecteds) = generate_batch(10, data_list, 0.25)
    print("real's shape: " + str(reals[0].shape))
    print("expected's shape: " + str(expecteds[0].shape))

    x = tf.placeholder(tf.float32, [None, 364, 512, 1])
    y_ = tf.placeholder(tf.float32, [None, 91, 128])

    y_conv = cnn_model_fn(x, y_)

    cost = tf.abs(y_conv - y_)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init) 

    print("------------------OUT SHAPES-------------------")
    print(y_.get_shape())
    print(y_conv.get_shape()) 
    print("-----------------------------------------------")

    while(True):
        (reals, expecteds) = generate_batch(10, data_list, 0.25)
        print("--------")
        #print("Cost before: ", sess.run(cost, feed_dict={x: reals, y_: expecteds}))
        train_step.run(session=sess, feed_dict={x: reals, y_: expecteds})
        #print("Cost after: ", sess.run(cost, feed_dict={x: reals, y_: expecteds}))
        print("--------")
        # see the first image
        outs = sess.run(y_conv, feed_dict={x: reals})
        print(len(outs))
        for i in range(len(reals)):
            cv2.imshow("input", reals[i])
            cv2.imshow("expected", expecteds[i])
            cv2.imshow("output", outs[i])
            if cv2.waitKey(200) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()

