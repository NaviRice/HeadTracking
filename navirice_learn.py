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


print_image_stats("./DATA")
    
