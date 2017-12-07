# from navirice_get_image import *
from navirice_helpers import navirice_image_to_np

import cv2
import numpy as np
import time

from navirice_get_image import *

DEFAULT_HOST = '127.0.0.1'  # The remote host
DEFAULT_PORT = 29000        # The same port as used by the server

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceCascade1 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
faceCascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
sideCascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
cascades = [faceCascade, faceCascade1, faceCascade2, sideCascade]


def get_head_from_img(np_image):
    """Detects head from image and returns location of it.

    returns:
        Tuple of x, y, and radius values, where domains are 0-1
        None (if head is not detected
    """
    image = _preprocess_image(np_image)
    potential_boxes = _run_cascades(image)

    if(len(potential_boxes) == 0):
        # Debug print RGB
        # cv2.imshow("rgb", image)
        return None

    (x, y, radius) = _get_head_from_boxes(image, potential_boxes)
    print("scaled: {}, {}, {}".format(x, y, radius))
    # Debug print RGB
    # cv2.imshow("rgb", image)

    return (x, y, radius)


def _preprocess_image(image):
    """Returns a smaller and grayscaled image. Smaller to faster process img, grayscale for Haar cascades."""
    resize_scale = 0.3
    resized_img = cv2.resize(
            image,None,fx=resize_scale, fy=resize_scale, interpolation = cv2.INTER_CUBIC)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    return gray_img

def _run_cascades(image):
    """Returns cv2.rectangle boxes of where it thinks heads are.

    For now it breaks after it finds the first cascade.
    """
    global cascades
    boxes = []
    for cascade in cascades:
        boxes = cascade.detectMultiScale(
            image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 34),
            flags = cv2.CASCADE_SCALE_IMAGE)
        if(len(boxes) != 0):
            break

    return boxes

def _get_head_from_boxes(image, boxes):
    """Expects a list of boxes, or will throw an error.

    Right now, it just takes the first box in the potential boxes (basically
     only counting for one haar cascade), but is left as a list for future
     optimization if needed.

    returns x, y, and radius from head detected boxes. Scaled to domains of 0-1
    """
    print("FACE DETECTED")
    image_height= image.shape[0]
    image_width = image.shape[1]

    (top_left_x, top_left_y, box_width, box_height) = boxes[0]
    x = top_left_x + box_width/2
    y = top_left_y + box_height/2
    radius = max(box_width, box_height)/2

    scaled_x = x/image_width
    scaled_y = y/image_height
    scaled_radius = max(box_width/image_width, box_height/image_height)/2

    # Debug draw circle/rectangle on face
    # cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), thickness=10, lineType=8, shift=0)
    #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return (scaled_x, scaled_y, scaled_radius)


def main():
    """Main to test this function. Should not be run for any other reason."""
    last_count = 0
    while(1):
        img_set, last_count = navirice_get_image(DEFAULT_HOST, DEFAULT_PORT, last_count)
        if(img_set != None):
            print("IR width: {}\nIR height: {}\nIR channels: {}\n".format(img_set.IR.width, img_set.IR.height, img_set.IR.channels))
            np_image = navirice_image_to_np(img_set.RGB)
            get_head_from_img(np_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()