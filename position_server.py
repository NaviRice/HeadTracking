import socket
import threading
import time
import struct

class PositionServer:
    def __init__(self, port):
        self.conn_addr = []
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
        print("VALUE SET")
        self.lock.acquire()
        self.x = x
        self.y = y
        self.z = z
        self.lock.release()
        self.distribute_all()

    def listener(self):
        print("LISTNER")
        while(True):
            conn, addr = self.s.accept()
            self.conn_addr.append((conn, addr))
            self.distribute_all()
            #d = threading.Thread(target=self.distributer, args=(conn, addr,))
            #d.daemon = True
            #d.start()

    def distribute_all(self):
        print("ALL DISTRIBUTOR")
        print("Current list:{}".format(self.conn_addr))
        self.lock.acquire()
        xbs = bytearray(struct.pack("f", self.x))
        ybs = bytearray(struct.pack("f", self.y))
        zbs = bytearray(struct.pack("f", self.z))
        self.lock.release()
        ab = xbs + ybs + zbs
        for conn, addr in self.conn_addr:
            try:
                conn.send(ab)
            except:
                # Remove connection it from the list
                self.conn_addr = [x for x in self.conn_addr if x[0] != conn]
                print(addr, " -closed")
                conn.close()
