from navirice_helpers import navirice_image_to_np
from navirice_helpers import navirice_ir_to_np
import cv2
import numpy as np
import navirice_image_pb2

from time import sleep


def main():
    count = 0
    # mode = "ir_dance_"
    # mode = "All_"
    # mode = "head_"
    # mode = "nohead_"
    mode = "default_"
    while True:
        try: 
            f = open("DATA/" + mode + str(count) + ".img_set", "rb")
            data = f.read()
            img_set = navirice_image_pb2.ProtoImageSet()
            img_set.ParseFromString(data)
            depth_image = navirice_image_to_np(img_set.Depth)
            IR_image = navirice_ir_to_np (img_set.IR)
            # cv2.imshow("Hi Can: count: " + str(count), depth_image)
            cv2.imshow("Depth", depth_image)
            cv2.imshow("IR", IR_image)
            cv2.waitKey(50)
        except FileNotFoundError:
            print("Stop asking!")
            pass
        count += 1


if __name__ == "__main__":
    main()
