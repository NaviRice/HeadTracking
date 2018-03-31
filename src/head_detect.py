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
np_img_stack = []

queue_mutex = Lock()
stack_mutex = Lock()

# namedtuples can be treated as small immutable classes
# Numpy image set, similar to protobuf img_set except with numpy images
Np_img_set = namedtuple('Np_img_set', 'img_count ir depth')
HeadData = namedtuple('HeadData', 'x y radius depth')
DetectedHead = namedtuple('DetectedHead', 'img_count head_data')


def main():
    kinect_client = _get_kinect_client("real")
    position_server = PositionServer(4007)
    single_thread_main(kinect_client, position_server)
    # multithreaded_main(kinect_client, position_server)


def single_thread_main(kinect_client, position_server):
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
        start_latency = time.time()
        np_img_set = get_np_img_set(kinect_client)
        if np_img_set is None:
            continue
        potential_head_data = get_detected_head(np_img_set, cascades)
        if potential_head_data is None:
            # cv2.imshow("herromyfriend", np_img_set.ir)
            # cv2.waitKey(1)
            continue
        end_latency = time.time()
        print(str((end_latency - start_latency)* 1000) + " ms")
        head_data = potential_head_data
        # cv2.imshow("depth", np_img_set.depth)
        draw_circle(np_img_set.ir, head_data.x, head_data.y, head_data.radius)
        render_head_data = _calculate_render_info(head_data)
        position_server.set_values(
                render_head_data.x, render_head_data.y, render_head_data.depth)
        # print("Render data: {} {} {}".format(render_head_data.x, render_head_data.y, render_head_data.depth))
        # print("depth data: {2}".format(render_head_data.x, render_head_data.y, render_head_data.depth))
        # prev_head_data = head_data


def multithreaded_main(kinect_client, position_server):
    initialize_haar_threads(thread_count=1)
    while(1):
        np_img_set = get_np_img_set(kinect_client)
        add_new_img_set_to_stack(np_img_set)
        handle_detected_heads(np_img_set, position_server)


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
    for _ in range(thread_count):
        thread = Thread(target=haar_thread)
        thread.daemon = True
        thread.start()


def haar_thread():
    """Adds data to detected_heads_queue by running Haar cascades from the np_img_stack.
    Cascades must be created per thread."""
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
        # Todo: use count so that you only get newer np_img_sets
        stack_mutex.acquire()
        np_img_set = np_img_stack.pop() if len(np_img_stack) > 0 else None
        stack_mutex.release()
        if np_img_set is None:
            continue

        potential_head_data = get_detected_head(np_img_set, cascades)
        if potential_head_data is None:
            continue
        head_data = potential_head_data

        detected_head = DetectedHead(np_img_set.img_count, head_data)
        queue_mutex.acquire()
        detected_heads_queue.append(detected_head)
        queue_mutex.release()


def get_detected_head(np_img_set, cascades):
    """Detects a head and returns HeadData depending on input.

    Head data is scaled normalized, x, y, and radius go from 0 to 1.
    depth is same as raw data"""
    potential_boxes = _run_cascades(np_img_set.ir, cascades)

    if(len(potential_boxes) == 0):
        return None
    boxes = potential_boxes

    head_data = _get_head_data_from_boxes(np_img_set, boxes)
    return head_data


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


def _get_head_data_from_boxes(np_img_set, boxes):
    """Expects a list of boxes, or will throw an error.

    Right now, it just takes the first box in the potential boxes (basically
     only counting for one haar cascade), but is left as a list for future
     optimization if needed.

    returns a HeadData tuple (includes x, y, radius, and depth). Scaled to domains of 0-1
    """

    # Get center of box
    (top_left_x, top_left_y, box_width, box_height) = boxes[0]
    x = top_left_x + box_width/2
    y = top_left_y + box_height/2
    radius = max(box_width, box_height)/2
    # y and x are flipped, according to kinect readings
    depth = float(np_img_set.depth[int(y), int(x)])

    head_data = HeadData(x, y, radius, depth)
    head_data = normalize_head_data(np_img_set.ir, box_width, box_height, head_data)

    return head_data 

def normalize_head_data(image, box_width, box_height, head_data):
    """Change head data from whatever it's scale to 0-1 for x, y, and radius.
    Depth stays the same."""
    image_height = image.shape[0]
    image_width = image.shape[1]
    x = head_data.x/image_width
    y = head_data.y/image_height
    radius = max(box_width/image_width, box_height/image_height)/2
    depth = head_data.depth # Depth should not scale
    head_data = HeadData(x, y, radius, depth)
    return head_data


def get_np_img_set(kinect_client):
    """Returns a img set of ir and depth iamges as numpy arrays from kinect_client.
    
    Also uses kinect_clients last_count to keep track of what image order."""
    img_set, last_count = kinect_client.navirice_get_next_image()
    np_ir_image = navirice_ir_to_np(img_set.IR)
    np_depth_image = navirice_image_to_np(img_set.Depth)
    # Todo check if depth is actually ir rn.
    (height, width, _) = np_ir_image.shape

    # Quick Scaling Down
    scale_value = 0.5
    width = int(width * scale_value)
    height = int(height * scale_value)
    np_ir_image = cv2.resize(np_ir_image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
    np_depth_image = cv2.resize(np_depth_image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)

    
    # Quick Cropping
    # height = int((1.0/3.0) * height)
    # width = int(0.5 * width)
    # np_ir_image = np_ir_image[height:, :width]
    # np_depth_image = np_depth_image[height:, :width]


    # Todo map the values of ir max/min after scaling and cropping.

    np_img_set = Np_img_set(last_count, np_ir_image, np_depth_image)
    return np_img_set


def add_new_img_set_to_stack(np_image_set):
    # Do in need global here? Todo
    stack_mutex.acquire()
    np_img_stack.append(np_image_set)
    stack_mutex.release()

# prev_head_data = HeadData(0, 0, 0, 0)
def handle_detected_heads(np_img_set, position_server):
    """Read data in from global detected_heads_queue if any and does something
    with it based on inputs."""
    global prev_head_data
    queue_mutex.acquire()
    global detected_heads_queue # Do I need global here?
    if len(detected_heads_queue) == 0: # Draw image without circle if none detected
        cv2.imshow("herromyfriend", np_img_set.ir)
        cv2.waitKey(1)
        # draw_circle(np_img_set.ir, prev_head_data.x, prev_head_data.y, prev_head_data.radius)

    while(len(detected_heads_queue) != 0):
        detected_head = detected_heads_queue.pop(0)
        # possibly loose data but may decrease latency:
        # detected_head = detected_heads_queue.pop()
        # detected_heads_queue = []
        # Not convinced that that actually matters though due to quick loop that
        # quickly displays things in order.
        head_data = detected_head.head_data
        draw_circle(np_img_set.ir, head_data.x, head_data.y, head_data.radius)
        render_head_data = _calculate_render_info(head_data)
        position_server.set_values(
                render_head_data.x, render_head_data.y, render_head_data.depth)
        # prev_head_data = head_data
    queue_mutex.release()


def draw_circle(np_image, x, y, radius):
    image_height= np_image.shape[0]
    image_width = np_image.shape[1]
    cv2.circle(np_image, (int(x*image_width), int(y*image_height)),
            int(radius*image_width), (255, 255, 255), thickness=5,
            lineType=8, shift=0)
    cv2.imshow("herromyfriend", np_image)
    cv2.waitKey(1)


def _calculate_render_info(head_data):
    """Reformats x and y from 0 to 1 range to -1 to 1 range."""
    x, y, = head_data.x, head_data.y
    x_render = -((x * 2) - 1)
    y_render  = -((y * 2) - 1)
    depth = head_data.depth * 5000
    head_data = HeadData(x_render, y_render, head_data.radius, depth)
    return head_data


if __name__ == "__main__":
    main()

# Legacy code potentially used by machine learning portion of the project
def get_head_from_image():
    # Todo: Copy from old code before refactor
    pass
    