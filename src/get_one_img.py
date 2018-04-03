from kinect.kinect_client import KinectClient
from kinect.fake_kinect_client import FakeKinectClient
from navirice_helpers import navirice_image_to_np
from navirice_helpers import navirice_ir_to_np


DEFAULT_HOST = "192.168.1.129"
DEFAULT_PORT = 29000

def _get_kinect_client(kinect_type="real"):
    """Gives back fake kinect client if kinect_type is fake, otherwise 
    give real one.
    If you don't have a physical kinect with you, use fake kinect client"""
    if kinect_type is "fake":
        kinect_client = FakeKinectClient(DEFAULT_HOST, DEFAULT_PORT)
    else:
        kinect_client = KinectClient(DEFAULT_HOST, DEFAULT_PORT)
    kinect_client.navirice_capture_settings(rgb=False, ir=True, depth=True)
    return kinect_client


kinect_client = _get_kinect_client(kinect_type="fake")

def get_depth_and_ir_from_kinect():
    img_set, last_count = kinect_client.navirice_get_next_image()

    np_depth_image = navirice_image_to_np(img_set.Depth, scale=False)
    np_ir_image = navirice_image_to_np(img_set.IR, scale=False)
    return (np_depth_image, np_ir_image)


# (np_depth_image, np_ir_image) = get_depth_and_ir_from_kinect()
# print(np_depth_image)
