# Echo client program
import socket
import time
import os

os.system("protoc -I=./proto --python_out=. ./proto/navirice_image.proto")

import navirice_image_pb2

class Image:
    def __init__(self, width, height, channels, type_, data, data_size):
        self.width = width
        self.height = height
        self.channels = channels
        self.type_ = type_
        self.data = data
        self.data_size = data_size

class KinectClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((host, port))
        self.last_count = 0

    def reconnect(self):
        self.s.close()
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))

    def navirice_get_image(self):
        print("---Requesting new image...")
        request_msg = navirice_image_pb2.ProtoRequest()
        request_msg.state = navirice_image_pb2.ProtoRequest.IMAGE
        request_msg.count = 1
        bytes_sent = self.s.send(request_msg.SerializeToString())

        count_msg = self.s.recv(1024)
        count_obj = navirice_image_pb2.ProtoImageCount()
        count_obj.ParseFromString(count_msg)
        count = count_obj.count
        print("image count: ", count)

        continue_msg = navirice_image_pb2.ProtoAcknowledge()
        continue_msg.count = 1
        if self.last_count >= count:
            print("Requesting stop because image count not new")
            continue_msg.state = navirice_image_pb2.ProtoAcknowledge.STOP
            bytes_sent = self.s.send(continue_msg.SerializeToString())
            return None, self.last_count
        else:
            print("Requesting --continue")
            continue_msg.state = navirice_image_pb2.ProtoAcknowledge.CONTINUE
            bytes_sent = self.s.send(continue_msg.SerializeToString())

        data = "".encode()
        b_size = count_obj.byte_count
        print("going to receive ", b_size, " bytes")
        t = self.s.recv(b_size, socket.MSG_WAITALL)
        data += t
        print("received total of ", len(data), " bytes")
        img_set = navirice_image_pb2.ProtoImageSet()
        img_set.ParseFromString(data)
        self.last_count = count
        return img_set, count
