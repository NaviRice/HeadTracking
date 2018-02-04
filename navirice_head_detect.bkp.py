# from navirice_get_image import *
from navirice_helpers import navirice_image_to_np

import cv2
import numpy as np
import time

from navirice_get_image import *

DEFAULT_HOST = '192.168.1.31'  # The remote host
DEFAULT_PORT = 29000        # The same port as used by the server

#noseCascade = cv2.CascadeClassifier("./Nariz.xml")
#noseCascade = cv2.CascadeClassifier("./nose.xml")
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceCascade1 = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
faceCascade2 = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
sideCascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
#noseCascade = cv2.CascadeClassifier("./righteye.xml")

def process_rgb(image, cascades):
    print(np.amax(image))
    image = cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # infrared
    #gray_img = np.array(image, dtype = np.uint8)

    for cascade in cascades:
        boxes = cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(20, 34), flags = cv2.CASCADE_SCALE_IMAGE)
        #noses = cascade.detectMultiScale(gray_img, 1.3, 5)
        for (x, y, w, h) in boxes:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            exit

    cv2.imshow("rgb", image)
    #k = cv2.waitKey(0)

last_count = 0
cascades = [faceCascade, faceCascade1, faceCascade2, sideCascade]
while(1):
    img_set, last_count = navirice_get_image(DEFAULT_HOST, DEFAULT_PORT, last_count)
    if(img_set != None):
        print("IR width: {}\nIR height: {}\nIR channels: {}\n".format(img_set.IR.width, img_set.IR.height, img_set.IR.channels))
        im = navirice_image_to_np(img_set.Depth)

        process_rgb(im, cascades)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
