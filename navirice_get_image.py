# Echo client program
import socket
import time
import os
import numpy as np

os.system("protoc -I=. --python_out=. navirice_image.proto")

import navirice_image_pb2

def naviriceImageToNp(img):
    tp = img.data_type
    divisor = 1
    if(tp == navirice_image_pb2.ProtoImage.FLOAT):
        tp = np.float32
        img.channels = 1
        divisor = 4500.0
    else:
        tp = np.uint8
    rgb_raw = np.frombuffer(img.data, dtype=tp, count=img.width*img.height*img.channels)
    if(img.data_type == navirice_image_pb2.ProtoImage.FLOAT):
        rgb_raw = (rgb_raw) / divisor
    im = rgb_raw.reshape((img.height, img.width, img.channels))
    return im



def navirice_get_image(host, port, last_count):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    count_msg = s.recv(1024);
    count_obj = navirice_image_pb2.ProtoImageCount()
    count_obj.ParseFromString(count_msg)
    count = count_obj.count
    if(last_count >= count):
        s.close()
        return None, last_count;

    continue_msg = navirice_image_pb2.ProtoAcknowledge()
    continue_msg.state = navirice_image_pb2.ProtoAcknowledge.CONTINUE
    bytes_sent = s.send(continue_msg.SerializeToString())

    data = "".encode()
    b_size = 10000000
    while(True):
        t = s.recv(b_size)
        if not t:
            break
        data += t
    s.close()
    img_set = navirice_image_pb2.ProtoImageSet()
    img_set.ParseFromString(data)
    return img_set, count;


