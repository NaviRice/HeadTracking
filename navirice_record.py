from navirice_get_image import KinectClient
from navirice_head_detect import get_head_from_img
from navirice_helpers import navirice_image_to_np
from navirice_helpers import navirice_img_set_write_file
from navirice_helpers import navirice_ir_to_np
import navirice_image_pb2

from tkinter import *

import cv2
import numpy as np
from threading import Thread
from enum import Enum

DEFAULT_HOST = '127.0.0.1'  # The remote host
DEFAULT_PORT = 29000        # The same port as used by the server

#todo
#  create a button which only records images if it detects a head
#  create a button which does not record images if it detects a head
#  Modify the protobuf image so that it also includes the information
#    for the label data. And record label data when recording information.

class State(Enum):
    STOP = 1
    RECORDALL = 2
    RECORDHEAD = 3
    RECORDNOHEAD = 4

class Window(Frame):
    """This program is kindof weird. The way it works is that a
    current_state var is modified, and depending on which one is
    chosen, a thread will modify the way it saves images."""

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.kinect_client = KinectClient(DEFAULT_HOST, DEFAULT_PORT)
        self.current_state = State.STOP
        self.master = master
        self.init_window()

    def init_window(self):
        self.should_pull = True
        self.should_run = True
        self.session_name = "default"
        self.last_count = 0
        self.master.title("NAVIRICE_RECORDER")

        recordBtn = Button(self, text="RECORD/Pause", command=self.recordAll)
        recordBtn.place(x=5, y=0)

        recordBtnOnlyHead = Button(
                self, text="RECORDOnlyHead", command=self.recordHead)
        recordBtnOnlyHead.place(x=5, y=30)

        recordBtnNoHead = Button(
                self, text="RECORDNoHead", command=self.recordNoHead)
        recordBtnNoHead.place(x=5, y=60)

        self.session_text = Text(self, height=1, width=20)
        self.session_text.place(x=5, y=90)
        self.session_text.insert(END, self.session_name)

        self.canvas = Canvas(self, height=60, width=120)
        self.print_states()
        self.pack(fill=BOTH, expand=1)

        thread = Thread(target = self.thread_stream)
        thread.start()

    def print_states(self):
        self.canvas.delete()
        fill = '#f00'
        if(self.current_state is not State.STOP):
            fill = '#0f0'
        self.canvas.create_oval(4, 0, 25, 25, outline="#000", fill=fill)
        self.canvas.pack(fill=BOTH, expand=1)
        self.canvas.place(x = 160, y = 0)

    def recordAll(self):
        """Records all images regardless of where head is."""
        # Todo: need to properly flip boolean here
        if self.current_state == State.RECORDALL:
            self.current_state = State.STOP
        else:
            self.current_state = State.RECORDALL
        self._record()

    def recordHead(self):
        """Records if head is detected."""
        if self.current_state == State.RECORDHEAD:
            self.current_state = State.STOP
        else:
            self.current_state = State.RECORDHEAD
        self._record()

    def recordNoHead(self):
        """Records if no head is detected."""
        if self.current_state == State.RECORDNOHEAD:
            self.current_state = State.STOP
        else:
            self.current_state = State.RECORDNOHEAD
        self._record()

    def _record(self):
        self.print_states()
        name = self.session_text.get("1.0",END)
        if(len(name)):
            self.session_name = name


    def kill(self):
        self.should_run = False

    def thread_stream(self):
        img_write_count = 0
        while(self.should_run):
            img_set = None
            if(self.should_pull):
                img_set, self.last_count = self.kinect_client.navirice_get_image()
            if(img_set is None):
                continue
            print("Current State: {}".format(self.current_state))
            if self.current_state is not State.STOP:
                img_write_count +=1
                _write_file(
                        self.session_name, img_set, img_write_count,
                        self.current_state)
            #cv2.imshow("RGB", navirice_image_to_np(img_set.RGB))
            cv2.imshow("DEPTH", navirice_image_to_np(img_set.Depth))
            cv2.imshow("IR", navirice_ir_to_np(img_set.IR))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("q pressed in cv window")
                return
            del img_set


def _write_file(session_name, img_set, count, current_state):
    """Saves image, and filters it based on current_state and whether or not
    a head has been detected."""
    if current_state is State.STOP:
        return
    if current_state is State.RECORDALL:
        navirice_img_set_write_file(session_name, img_set, count)
        return

    np_image = navirice_ir_to_np(img_set.IR)
    head_location = get_head_from_img(np_image)

    if current_state is State.RECORDHEAD and head_location is not None:
        navirice_img_set_write_file(session_name, img_set, count)
    elif current_state is State.RECORDNOHEAD and head_location is None:
        navirice_img_set_write_file(session_name, img_set, count)


def main():
    # count = 0
    # kinect_client = KinectClient(DEFAULT_HOST, DEFAULT_PORT)
    # while(count < 1000):
    #     img_set, last_count = kinect_client.navirice_get_image()
    #     if img_set is not None:
    #         last_count = navirice_img_set_write_file("irdance", img_set, count)
    #         count += 1
    root = Tk()
    root.geometry("190x140")
    root.attributes('-type', 'dialog')
    app = Window(root)
    def on_quit():
        app.kill()
        exit()
    root.protocol("WM_DELETE_WINDOW", on_quit)
    root.mainloop()

if __name__ == "__main__":
    main()

