from navirice_get_image import KinectClient
from FakeKinect import FakeKinectClient

from navirice_helpers import navirice_image_to_np
from navirice_head_detect import get_head_from_img

import cv2
import numpy as np
import time

DEFAULT_HOST = '127.0.0.1'  # The remote host
DEFAULT_PORT = 29000        # The same port as used by the server


def main():
    kinect_client = KinectClient(DEFAULT_HOST, DEFAULT_PORT)
    kinect_client.navirice_capture_settings(rgb=False, ir=True, depth=True)
    # kinect_client = FakeKinectClient(DEFAULT_HOST, DEFAULT_PORT)
    last_count = 0

    first_img_set, last_count = _get_next_image(kinect_client)
    raw_bg_depth = navirice_image_to_np(first_img_set.Depth) - 0.05
    background_depth = np.where(raw_bg_depth<=0.0, 1, raw_bg_depth)
    cv2.imshow("Background Depth", background_depth) # show preview
    if cv2.waitKey(1) & 0xFF == ord('q'):
        return

    while(1):
        img_set, last_count = _get_next_image(kinect_client)
        current_depth = navirice_image_to_np(img_set.Depth)
        current_depth = np.where(current_depth<=0.0, 1, current_depth)
        current_ir = navirice_image_to_np(img_set.IR)
        forground_depth, forground_ir = _extract_forground(
                background_depth, current_depth, current_ir)

        cv2.imshow("Forground Depth", forground_depth) # show preview
        cv2.imshow("Forground Ir", forground_ir) # show preview

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def _get_next_image(kinect_client):
    img_set = None
    last_count = - 1
    while (img_set == None
            or img_set.IR.width == 0
            or img_set.Depth.width == 0
            or last_count == -1):
        img_set, last_count = kinect_client.navirice_get_image()
    return img_set, last_count


def _extract_forground(background_depth, current_depth, current_ir):
    width = current_depth.shape[1]
    height = current_depth.shape[0]
    mask = current_depth >= background_depth
    current_depth[mask] = 0
    current_ir[mask] = 0
    return current_depth, current_ir


if __name__ == "__main__":
    main()

