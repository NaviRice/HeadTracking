from navirice_get_image import KinectClient
from navirice_helpers import navirice_img_set_write_file
from navirice_helpers import navirice_image_to_np
from navirice_helpers import navirice_ir_to_np
import navirice_image_pb2

from tkinter import *

import cv2
import numpy as np
from threading import Thread

HOST = '127.0.0.1'  # The remote host
PORT = 29000        # The same port as used by the server


class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    def init_window(self):
        self.should_pull = True
        self.should_record = False
        self.should_run = True
        self.session_name = "default"
        self.last_count = 0
        self.master.title("NAVIRICE_RECORDER")

        recordButton = Button(self, text="RECORD", command=self.record)
        recordButton.place(x=5, y=0)

        self.session_text = Text(self, height=1, width=20)
        self.session_text.place(x=5, y=30)
        self.session_text.insert(END, self.session_name)

        self.canvas = Canvas(self, height=30, width=30)
        self.print_states()
        self.pack(fill=BOTH, expand=1)

        thread = Thread(target = self.thread_stream)
        thread.deamon = True
        thread.start()

    def print_states(self):
        self.canvas.delete()
        fill = '#f00'
        if(self.should_record):
            fill = '#0f0'
        self.canvas.create_oval(4, 0, 25, 25, outline="#000", fill=fill)
        self.canvas.pack(fill=BOTH, expand=1)
        self.canvas.place(x = 100, y = 0)

    def record(self):
        self.should_record = not self.should_record
        self.print_states()
        name = self.session_text.get("1.0",END)
        if(len(name)):
            self.session_name = name

    def kill(self):
        self.should_run = False

    def thread_stream(self):
        kc = KinectClient(HOST, PORT)
        kc.navirice_capture_settings(False, True, True)
	
        while(self.should_run):
            img_set = None

            if(self.should_pull):
                img_set, self.last_count = kc.navirice_get_image()
                if(img_set != None and img_set.IR.width > 0 and img_set.Depth.width > 0):
                    if self.should_record:
                        #processThread =Thread(target=navirice_img_set_write_file, args=[self.session_name, img_set, self.last_count])
                        #processThread.start()
                        navirice_img_set_write_file(self.session_name, img_set, self.last_count)
                    cv2.imshow("IR", navirice_ir_to_np(img_set.IR))
                    cv2.imshow("DEPTH", navirice_image_to_np(img_set.Depth))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("q pressed in cv window")
                    del img_set


def main():
    root = Tk()
    root.geometry("170x65")
    root.attributes('-type', 'dialog')
    app = Window(root)
    def on_quit():
        app.kill()
        exit()
    root.protocol("WM_DELETE_WINDOW", on_quit)
    root.mainloop()

if __name__ == "__main__":
    main()

