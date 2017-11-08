from navirice_get_image import *

import cv2
import numpy as np

DEFAULT_HOST = '127.0.0.1'  # The remote host
DEFAULT_PORT = 29000        # The same port as used by the server

last_count = 0
while(1):
    img_set, last_count = navirice_get_image(DEFAULT_HOST, DEFAULT_PORT, last_count)
    if(img_set != None):
        print("IMG#: ", img_set.count)
        print("Depth width: ", img_set.Depth.width)
        print("Depth height: ", img_set.Depth.height)
        print("Depth channels: ", img_set.Depth.channels)
        rgb_raw = np.frombuffer(img_set.Depth.data, dtype=np.float32, count=img_set.Depth.width*img_set.Depth.height*1)
        rgb_raw = (rgb_raw/4500)
        im = rgb_raw.reshape((img_set.Depth.height, img_set.Depth.width, 1))
        cv2.imshow('', im)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

