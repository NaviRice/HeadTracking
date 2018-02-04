from navirice_get_image import KinectClient
from navirice_helpers import navirice_image_to_np
from navirice_helpers import navirice_ir_to_np

from threading import Thread, Lock
import cv2
import numpy as np
import time

DEFAULT_HOST = '127.0.0.1'  # The remote host
DEFAULT_PORT = 29000        # The same port as used by the server

detected_heads_queue = []
stack = []

queue_mutex = Lock()
stack_mutex = Lock()

class QueueHead():

    def __init__(self, count, head):
        """Head is (x, y, radius)."""
        self.count = count
        self.head = head


def smart_threaded_haar_test():
    global detected_heads_queue

    kinect_client = KinectClient(DEFAULT_HOST, DEFAULT_PORT)
    last_count = 0
    for i in range(2):
        thread = Thread(target=thread_worker)
        thread.daemon = True
        thread.start()

    while(1):
        img_set, last_count = kinect_client.navirice_get_image()
        if(img_set != None):
            np_image = navirice_ir_to_np(img_set.IR)
            stack_mutex.acquire()
            print("ADDING IMAGE TO STACK")
            stack.append(np_image)
            stack_mutex.release()

            queue_mutex.acquire()
            for item in detected_heads_queue:
                head = item.head
                if item.count > 0:
                    image_height= np_image.shape[0]
                    image_width = np_image.shape[1]
                    x, y, radius = head[0], head[1], head[2]
                    cv2.circle(np_image, (int(x*image_width), int(y*image_height)),
                            int(radius*image_width), (255, 255, 255), thickness=5,
                            lineType=8, shift=0)
                    print("OHMG: {}, {}, {}".format(x, y, radius))
                    item.count -= 1
            for item in detected_heads_queue:
                if item.count <= 0:
                    detected_heads_queue.remove(item)
            queue_mutex.release()
            cv2.imshow("Show", np_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

def thread_worker():
    print("THREAD WORKKERKERKTRJ!!!")
    faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faceCascade1 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    faceCascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
    sideCascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
    cascades = [faceCascade, faceCascade1, faceCascade2, sideCascade]
    while True:
        img = None
        stack_mutex.acquire()
        if len(stack) > 0:
            img = stack.pop()
            print("REMOVING IMAGE FROM STACK")
        stack_mutex.release()
        if img is not None:
            print("PROCESSING IMAGE")
            # potential_location is (x, y, radius)
            potential_location = get_head_from_img(img, cascades)
            if potential_location is not None:
                queue_mutex.acquire()
                queue_head = QueueHead(3, potential_location)
                detected_heads_queue.append(queue_head)
                queue_mutex.release()
    print("SO DED!!!")


def get_head_from_img(np_image, cascades):
    """Detects head from image and returns location of it.

    returns:
        Tuple of x, y, and radius values, where domains are 0-1
        None (if head is not detected
    """

    #i dont bother scaling and graying the image because... its 500 pixels wide and already grayscale
#    image = _preprocess_image(np_image)
    image = np_image
    potential_boxes = _run_cascades(image, cascades)

    if(len(potential_boxes) == 0):
        # cv2.imshow("ir", image)
        return None

    (x, y, radius) = _get_head_from_boxes(image, potential_boxes)
    print("scaled: {}, {}, {}".format(x, y, radius))
    # Debug show image ir
    #cv2.imshow("ir", image)

    return (x, y, radius)


def _preprocess_image(image):
    """Returns a smaller and grayscaled image. Smaller to faster process img, grayscale for Haar cascades.
    Deprecated. Used for Harr Cascades on RGB images. We do Harr Cascades on IR now."""
    resize_scale = 0.3
    resized_img = cv2.resize(
            image,None,fx=resize_scale, fy=resize_scale, interpolation = cv2.INTER_CUBIC)
    gray_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    return gray_img

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
    # cv2.circle(image, (int(x), int(y)), int(radius), (255, 255, 255), thickness=10, lineType=8, shift=0)
#    cv2.rectangle(image, (x, y), (x+box_width, int(y+box_height)), (255, 255, 255), 2)

    return (scaled_x, scaled_y, scaled_radius)

def main():
    """Main to test this function. Should not be run for any other reason."""
    smart_threaded_haar_test()
    #kinect_head_detect_test()
    #send_head_data_to_rendering_server()


def kinect_head_detect_test():
    kinect_client = KinectClient(DEFAULT_HOST, DEFAULT_PORT)
    last_count = 0
    while(1):
        img_set, last_count = kinect_client.navirice_get_image()
        if(img_set != None):
            print("IR width: {}\nIR height: {}\nIR channels: {}\n".format(
                img_set.IR.width, img_set.IR.height, img_set.IR.channels))
            np_image = navirice_ir_to_np(img_set.IR)
            get_head_from_img(np_image)
            cv2.imshow("IR", np_image) # show preview
            #cv2.imshow("IR", cv2.resize(np_image,None,fx=2.0, fy=2.0, interpolation = cv2.INTER_CUBIC)) #show preview, but bigger

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def send_head_data_to_rendering_server():
    last_count = 0
    while(1):
        img_set, last_count = kinect_client.navirice_get_image()
        if(img_set != None):
            print("IR width: {}\nIR height: {}\nIR channels: {}\n".format(
                img_set.IR.width, img_set.IR.height, img_set.IR.channels))
            np_image = navirice_ir_to_np(img_set.IR)
            get_head_from_img(np_image)
            cv2.imshow("IR", np_image) # show preview
            #cv2.imshow("IR", cv2.resize(np_image,None,fx=2.0, fy=2.0, interpolation = cv2.INTER_CUBIC)) #show preview, but bigger

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    main()
