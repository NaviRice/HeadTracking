import settings
from kinect.kinect_client import KinectClient
from kinect.fake_kinect_client import FakeKinectClient
from position_server.position_server import PositionServer
from navirice_helpers import navirice_image_to_np
from navirice_helpers import navirice_ir_to_np

from threading import Thread, Lock
from collections import namedtuple
import cv2
import numpy as np
import time

DEFAULT_HOST = '127.0.0.1'  # The remote host
DEFAULT_PORT = 29000        # The same port as used by the server

detected_heads_queue = []
stack = []

queue_mutex = Lock()
stack_mutex = Lock()

HeadPos = namedtuple('HeadPos', 'x y radius')

class QueueHead():

    def __init__(self, count, head_pos):
        """Head is a named tuple HeadPos (x, y, radius)."""
        self.count = count
        self.head_pos = head_pos


def main():
    """Main to test this function. Should not be run for any other reason."""
    kinect_client = _get_kinect_client("real")
    position_server = PositionServer(4007)
    initialize_haar_threads(thread_count=2)
    while(1):
        np_ir_image, np_depth_image = get_ir_and_depth_imgs(kinect_client)
        add_new_img_to_stack(np_ir_image)
        head_pos = handle_detected_heads(np_ir_image, np_depth_image, position_server)


def mediator(kinect_type="real", data_to_renderer=True, threaded=False, ):
    """Based on given parameters, will run the program.

    The following are the default values
    kinect_type="real"      Requires physical kinect to be connected.
        can also be "fake"
    data_to_renderer=True   Creates PositionalServer that sends head data.
    threaded=False          Runs Haar Cascades in threads
    """
    pass




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


def initialize_haar_threads(thread_count=2):
    for i in range(thread_count):
        thread = Thread(target=thread_worker)
        thread.daemon = True
        thread.start()


def thread_worker():
    # print("Thread worker started")
    faceCascade = cv2.CascadeClassifier(
            settings.CASCADES_DIR + 'haarcascade_frontalface_default.xml')
    faceCascade1 = cv2.CascadeClassifier(
            settings.CASCADES_DIR + 'haarcascade_frontalface_alt.xml')
    faceCascade2 = cv2.CascadeClassifier(
            settings.CASCADES_DIR + 'haarcascade_frontalface_alt2.xml')
    sideCascade = cv2.CascadeClassifier(
            settings.CASCADES_DIR + 'haarcascade_profileface.xml')
    cascades = [faceCascade, faceCascade1, faceCascade2, sideCascade]
    while True:
        stack_mutex.acquire()
        img = stack.pop() if len(stack) > 0 else None
        stack_mutex.release()
        if img is None:
            continue
        # print("Grab image from stack")
        # potential_location is (x, y, radius)
        potential_location = get_head_from_img(img, cascades)
        if potential_location is None:
            continue
        queue_mutex.acquire()
        # print("Found head location put in queue head queue")
        queue_head = QueueHead(3, potential_location)
        detected_heads_queue.append(queue_head)
        queue_mutex.release()


def get_head_from_img(np_image, cascades, should_scale=True):
    """Detects head from image and returns location of it.

    returns:
        Tuple of x, y, and radius values, where domains are 0-1
        None (if head is not detected
    """
    # Since image should be ir, the data does not need to be grayscaled
    image = np_image
    potential_boxes = _run_cascades(image, cascades)

    if(len(potential_boxes) == 0):
        return None

    (x, y, radius) = _get_head_from_boxes(image, potential_boxes, should_scale)
    head_pos = HeadPos(x, y, radius)
    return head_pos


def _run_cascades(image, cascades):
    """Returns cv2.rectangle boxes of where it thinks heads are.

    For now it breaks after it finds the first cascade.
    """
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
    image_height= image.shape[0]
    image_width = image.shape[1]

    (top_left_x, top_left_y, box_width, box_height) = boxes[0]
    x = top_left_x + box_width/2
    y = top_left_y + box_height/2
    radius = max(box_width, box_height)/2

    if should_scale:
        x = x/image_width
        y = y/image_height
        radius = max(box_width/image_width, box_height/image_height)/2

    return (x, y, radius)


def get_ir_and_depth_imgs(kinect_client):
    """Returns ir and depth iamges as numpy arrays from kinect_client."""
    # print("Get image from kinect client")
    img_set, last_count = kinect_client.navirice_get_next_image()
    np_image = navirice_ir_to_np(img_set.IR)
    np_depth_image = navirice_image_to_np(img_set.Depth)
    return np_image, np_depth_image


def add_new_img_to_stack(np_image):
    stack_mutex.acquire()
    stack.append(np_image)
    stack_mutex.release()


def handle_detected_heads(np_ir_image, np_depth_image, position_server):
    """Read data in from global detected_heads_queue if any and does something
    with it based on inputs."""
    queue_mutex.acquire()
    global detected_heads_queue
    while(len(detected_heads_queue) != 0):
        head_pos = detected_heads_queue[0].head_pos
        detected_heads_queue.pop()
        draw_circle(np_ir_image, head_pos.x, head_pos.y, head_pos.radius)
        cv2.imshow("herromyfriend", np_ir_image)
        cv2.waitKey(1)
        notify_renderer(head_pos, np_depth_image, position_server)

    queue_mutex.release()


def draw_circle(np_image, x, y, radius):
    image_height= np_image.shape[0]
    image_width = np_image.shape[1]
    cv2.circle(np_image, (int(x*image_width), int(y*image_height)),
            int(radius*image_width), (255, 255, 255), thickness=5,
            lineType=8, shift=0)
    cv2.imshow("herromyfriend", np_image)
    cv2.waitKey(1)


def notify_renderer(head_data, np_depth_image, position_server):
    """
    Expects this to have already happened:
    position_server = PositionServer(40007)
    """
    (render_x, render_y, render_depth) = _calculate_render_info(
            np_depth_image, head_data)
    position_server.set_values(render_x, render_y, render_depth)


def _calculate_render_info(depth_image, head_pos):
    """Reformats x and y to range -1 to 1, and depth in centimeters."""
    x, y, radius = head_pos.x, head_pos.y, head_pos.radius
    image_height= depth_image.shape[0]
    image_width = depth_image.shape[1]
    x_render = (x/image_width * 2) - 1
    y_render  = -((y/image_height * 2) - 1)
    # y and x are flipped, according to kinect readings
    depth_render = float(depth_image[int(y), int(x)])
    return (x_render, y_render, depth_render)


if __name__ == "__main__":
    main()
