import socket
import threading
import time
import struct

HOST = '127.0.0.1'
PORT = 40007

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

while True:
    data = s.recv(12)
    x = struct.unpack('f', data[0:4])
    y = struct.unpack('f', data[4:8])
    z = struct.unpack('f', data[8:12])
    print("x: ", x, " y:", y, " z:", z)

