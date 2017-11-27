import numpy as np
import tensorflow as tf
import os
from navirice_get_image import * 
import random

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



# This means that we need an input layer that is 512x424 pixels
# with a 5x5x1 layer this makes the second layer 102x84 in size
# lets define the model:
def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features["x"], [-1, 512, 424, 1])

    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=102*84,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size=[2,2], strides = 2)
    
    print(pool1.get_shape())

def main():
    print_image_stats("./DATA")


if __name__ == "__main__":
    main()

