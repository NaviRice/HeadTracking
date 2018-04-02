import settings
from kinect.kinect_client import KinectClient
from kinect.fake_kinect_client import FakeKinectClient
from navirice_helpers import navirice_image_to_np
from navirice_helpers import navirice_ir_to_np

from threading import Thread, Lock
from collections import namedtuple
import cv2
import numpy as np
import time

#DEFAULT_HOST = '192.168.1.129'  # The remote host
DEFAULT_HOST = '127.0.0.1'  # The remote host
DEFAULT_PORT = 29000        # The same port as used by the server


# namedtuples can be treated as small immutable classes
# Numpy image set, similar to protobuf img_set except with numpy images
#Np_img_set = namedtuple('Np_img_set', 'img_count ir depth bg')


def main():
    kinect_client = _get_kinect_client("real")
    while True:
        img_set, last_count = kinect_client.navirice_get_next_image()
        np_ir_image = navirice_ir_to_np(img_set.IR)
        np_depth_image = navirice_image_to_np(img_set.Depth)
        np_bg_image = navirice_image_to_np(img_set.BG)
#        print(img_set.BG)
        (height, width, _) = np_ir_image.shape
        print(str(width) + " "+ str(height))
        np_bg_image = cv2.resize(np_bg_image, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
        draw_img(np_bg_image)
    # multithreaded_main(kinect_client, position_server)



def _get_kinect_client(kinect_type="real"):
    """Gives back fake kinect client if kinect_type is fake, otherwise
    give real one.
    If you don't have a physical kinect with you, use fake kinect client"""
    if kinect_type is "fake":
        kinect_client = FakeKinectClient(DEFAULT_HOST, DEFAULT_PORT)
    else:
        kinect_client = KinectClient(DEFAULT_HOST, DEFAULT_PORT)
    kinect_client.navirice_capture_settings(rgb=False, ir=True, depth=True, bg=True)
    return kinect_client



def draw_img(np_image):
    image_height= np_image.shape[0]
    image_width = np_image.shape[1]
##    cv2.circle(np_image, (int(x*image_width), int(y*image_height)),
##            int(radius*image_width), (255, 255, 255), thickness=5,
##            lineType=8, shift=0)
    cv2.imshow("herromyfriend", np_image)
    cv2.waitKey(1)



if __name__ == "__main__":
    main()
