from navirice_get_image import KinectClient
from FakeKinect import FakeKinectClient
from position_server import PositionServer
from navirice_helpers import navirice_image_to_np
from navirice_helpers import navirice_ir_to_np

import cv2
import numpy as np
import time

DEFAULT_HOST = '127.0.0.1'  # The remote host
DEFAULT_PORT = 29000        # The same port as used by the server

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceCascade1 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
faceCascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
sideCascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
cascades = [faceCascade, faceCascade1, faceCascade2, sideCascade]


def send_head_data_to_rendering_server():
    print("called")
    position_server = PositionServer(40007)
    # To run without physical kinect, type "fake", otherwise type "real"
    kinect_client = _get_kinect_client("real")
    kinect_client.navirice_capture_settings(rgb=False, ir=True, depth=True)
    last_count = 0
    while(1):
        img_set, last_count = kinect_client.navirice_get_image()
        if(img_set is None
                or img_set.IR.width == 0
                or img_set.Depth.width == 0):
            print("none image")
            continue

        np_ir_image = navirice_ir_to_np(img_set.IR)
        np_depth_image = navirice_image_to_np(img_set.Depth, scale=False)

        potential = get_head_from_img(np_ir_image, should_scale=False)
        if potential is None:
            continue
        head_location = potential

        # Debug Images
        # cv2.circle(np_ir_image, (0, 0), 5, (255, 255, 255),
                # thickness=10, lineType=8, shift=0)
        cv2.imshow("IR", np_ir_image) # show preview
        cv2.imshow("Depth", np_depth_image) # show preview

        (render_x, render_y, render_depth) = _calculate_render_info(
                np_depth_image, head_location)

        print("Render DATA: x:{}, y:{}, depth:{}".format(
            render_x, render_y, render_depth))
        position_server.set_values(render_x, render_y, render_depth)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def _get_kinect_client(kinect_type="real"):
    """Gives back fake kinect client if kinect_type is fake, otherwise 
    give real one.
    If you don't have a physical kinect with you, use fake kinect client"""
    if kinect_type is "fake":
        return FakeKinectClient(DEFAULT_HOST, DEFAULT_PORT)
    else:
        return KinectClient(DEFAULT_HOST, DEFAULT_PORT)


def _calculate_render_info(depth_image, head_location):
    """Returns -1 to 1 for x and y, and meters to head location for depth."""
    image_height= depth_image.shape[0]
    image_width = depth_image.shape[1]
    (x, y, radius) = head_location

    x_render = (x/image_width * 2) - 1
    y_render  = (y/image_height * 2) - 1

    # I think y and x should be flipped, according to kinect readings
    raw_depth_at_head = float(depth_image[int(y), int(x)])
    # Rendering server wants depth in meters
    # First time I get to write magic numbers yay!
    # Conversion taken from http://shiffman.net/p5/kinect/
    #depth_render = 1.0 / (raw_depth_at_head * -0.0030711016 + 3.3309495161);
    depth_render = raw_depth_at_head


    print("raw_data DATA: x:{}, y:{} raw:{} scaled:{}".format(
        x, y, raw_depth_at_head, depth_render))
    return (x_render, y_render, depth_render)


def get_head_from_img(np_image, should_scale=True):
    """Detects head from image and returns location of it.

    returns:
        Tuple of x, y, and radius values, where domains are 0-1
        None (if head is not detected
    """
    # Since image should be ir, the data does not need to be grayscaled
    image = np_image
    potential_boxes = _run_cascades(image)

    if(len(potential_boxes) == 0):
        # cv2.imshow("ir", image)
        return None

    (x, y, radius) = _get_head_from_boxes(image, potential_boxes, should_scale)
    # Debug show image ir
    #cv2.imshow("ir", image)

    return (x, y, radius)


def _run_cascades(image):
    """Returns cv2.rectangle boxes of where it thinks heads are.

    For now it breaks after it finds the first cascade.
    """
    global cascades
    boxes = []
    for cascade in cascades:
        boxes = cascade.detectMultiScale(
            image, scaleFactor = 1.1, minNeighbors=5, minSize=(20, 34),
            flags = cv2.CASCADE_SCALE_IMAGE)
        if(len(boxes) != 0):
            cascades.remove(cascade)
            cascades.insert(0, cascade)
            break

    return boxes


def _get_head_from_boxes(image, boxes, should_scale=True):
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

    # Debug draw circle/rectangle on face
    cv2.circle(image, (int(x), int(y)), 5, (255, 255, 255), thickness=10, lineType=8, shift=0)
    cv2.rectangle(
            image,
            (int(top_left_x), int(top_left_y)),
            (int(top_left_x + box_width), int(top_left_y + box_height)),
            (255, 255, 255), 2)
    if should_scale:
        x = x/image_width
        y = y/image_height
        radius = max(box_width/image_width, box_height/image_height)/2

    return (x, y, radius)



def kinect_head_detect_test():
    kinect_client = KinectClient(DEFAULT_HOST, DEFAULT_PORT)
    kinect_client.navirice_capture_settings(rgb=False, ir=True, depth=True)
    
    last_count = 0
    while(1):
        img_set, last_count = kinect_client.navirice_get_image()
        if(img_set != None
                and img_set.IR.width > 0
                and img_set.Depth.width > 0):
            print("IR width: {}\nIR height: {}\nIR channels: {}\n".format(
                img_set.IR.width, img_set.IR.height, img_set.IR.channels))
            np_image = navirice_ir_to_np(img_set.IR)
            get_head_from_img(np_image)
            cv2.imshow("IR", np_image) # show preview
            #cv2.imshow("IR", cv2.resize(np_image,None,fx=2.0, fy=2.0, interpolation = cv2.INTER_CUBIC)) #show preview, but bigger

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    """Main to test this function. Should not be run for any other reason."""
    # kinect_head_detect_test()
    send_head_data_to_rendering_server()


if __name__ == "__main__":
    main()
