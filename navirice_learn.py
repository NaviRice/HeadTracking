import numpy as np
import tensorflow as tf
import os
import navirice_image_pb2 
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

def generate_batch(count, data_list):
    i = 0
    real = []
    expected = []
    while i < count:
        ri = random.randint(0, len(data_list))
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
                    #real.append(tf.convert_to_tensor(depth_image, dtype=np.float32))
                    #expected.append(tf.convert_to_tensor(possible_bitmap, dtype=np.float32))
                    real.append(depth_image)
                    expected.append(possible_bitmap)
                    i += 1
    return (real, expected)




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
    data_list = load_data_file_list("./DATA")
    (reals, expecteds) = generate_batch(10, data_list)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    print("real's shape: " + str(reals[0].shape))
    print("expected's shape: " + str(expecteds[0].shape))


if __name__ == "__main__":
    main()

