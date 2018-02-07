import socket
import threading
import time
import struct

class PositionServer:
    def __init__(self, port):
        self.port = port
        self.host = ''
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind((self.host, self.port))
        self.s.listen(1)
        self.lock = threading.Lock()
        t = threading.Thread(target=self.listener)
        self.x = 0.0
        self.y = 1.0
        self.z = 0.5
        t.daemon = True
        t.start()
    
    def set_values(self, x, y ,z):
        self.lock.acquire()
        self.x = x
        self.y = y
        self.z = z
        self.lock.release()
    
    def listener(self):
        while(True):
            conn, addr = self.s.accept()
            d = threading.Thread(target=self.distributer, args=(conn, addr,))
            d.daemon = True
            d.start()

    def distributer(self, conn, addr):
        print(addr, " -connected")
        while(True):
            self.lock.acquire()
            xbs = bytearray(struct.pack("f", self.x))
            ybs = bytearray(struct.pack("f", self.y))
            zbs = bytearray(struct.pack("f", self.z))
            self.lock.release()
            ab = xbs + ybs + zbs 
            try:
                conn.send(ab)
            except:
                print(addr, " -closed")
                conn.close()
                break
