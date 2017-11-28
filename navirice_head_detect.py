# from navirice_get_image import *

import cv2
import numpy as np
import time

from navirice_get_image import *

DEFAULT_HOST = '127.0.0.1'  # The remote host
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
    #image = cv2.resize(image,None,fx=0.3, fy=0.3, interpolation = cv2.INTER_CUBIC)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
    print("Waat")
    img_set, last_count = navirice_get_image(DEFAULT_HOST, DEFAULT_PORT, last_count)
    if(img_set != None):
        print("RGB width: {}\nRGB height: {}\nRGB channels: {}\n".format(img_set.RGB.width, img_set.RGB.height, img_set.RGB.channels))
        rgb_raw = np.frombuffer(img_set.RGB.data, dtype=np.uint8, count=img_set.RGB.width*img_set.RGB.height*img_set.RGB.channels)
        #rgb_raw = np.frombuffer(img_set.RGB.data, dtype=np.uintv2.IMREAD_COLOR)
        count=(img_set.RGB.width*img_set.RGB.height*img_set.RGB.channels) #    #rgb_raw = (rgb_raw/4500)
        im = rgb_raw.reshape((img_set.RGB.height, img_set.RGB.width, img_set.RGB.channels))
        img = cv2.resize(im, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

        process_rgb(img, cascades)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#for i in range(100):
#    print("LOOP: {}".format(i))
#    img = cv2.imread('allFace/beautiful_face_{}.jpg'.format(i), cv2.IMREAD_COLOR)
#    #cv2.imwrite("beautiful_face_{}.jpg".format(save_img_count), img)
#    #save_img_count += 1
#    process_rgb(img, faceCascade)
#    #cv2.imshow('', img)
#    #time.sleep(1000)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break
