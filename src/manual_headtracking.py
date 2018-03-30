#!/usr/bin/python
from position_server.position_server import PositionServer

from threading import Thread
from tkinter import *
import math
import signal
import sys

# Global vars that change the way canvas looks
canvas_border = 50
canvas_width = canvas_height = 500

mouse_scroll_force_update = True

time_interval = 33 #5 fps

def signal_handler(signal, frame):
        position_server.close_listen()
        exit()
def main():
    root = Tk()
    root.resizable(width=False, height=False)
    root.geometry("170x65")
    root.attributes('-type', 'dialog')
    app = Window(root)
    def on_quit():
        position_server.close_listen()
#        app.kill()
        exit()
    signal.signal(signal.SIGINT, signal_handler)
    root.protocol("WM_DELETE_WINDOW", on_quit)
    root.mainloop()
class Window(Frame):

    def __init__(self, master=None):
        master.minsize(
                width=canvas_width + 2*canvas_border,
                height=canvas_height + 2*canvas_border)
        Frame.__init__(self, master)
        self.master = master
        self.init_window()
        self.record = False

	#selected  (last place the mouse was clicked on
        self.s_x = 0
        self.s_y = 0
        self.s_depth = 300

	#current mouse positions
        self.m_x = 0
        self.m_y = 0
        self.m_depth = 0
        #last sent positions
        self.sent_x = 0
        self.sent_y = 0
        self.sent_depth = 0



    def init_window(self):
        self.master.title("NAVIRICE_HEADTRACKING_MANUAL")

        self.session_text = Text(self, height=1, width=20)
        self.session_text.place(x=5, y=30)
        self.session_text.insert(END, "test")


        self.canvas = Canvas(self,
                height=canvas_height, width=canvas_width, bd=canvas_border)
        self.pack(fill=BOTH, expand=1)
        self.canvas.pack()
        self.canvas.create_rectangle(canvas_border, canvas_border,
                canvas_width+canvas_border, canvas_height+canvas_border, fill="black")
        self.canvas.bind("<Motion>", self.mousemove)
        self.canvas.bind("<ButtonPress-1>", self.mouse1press)
        self.canvas.bind("<ButtonRelease-1>", self.mouse1release)
        self.canvas.bind("<Button-4>", self.mousescrollup)
        self.canvas.bind("<Button-5>", self.mousescrolldown)
        self.canvas.after(time_interval, self.interval)

    def mousemove(self, event):
        self.m_x, self.m_y, self.m_depth = self.clip(event.x, event.y, self.m_depth)
        if self.record is True:
            self.s_x = self.m_x
            self.s_y = self.m_y
            self.s_depth = self.m_depth
        self.draw_stuff()

    def mouse1press(self, event):
        self.record = True
        self.m_x, self.m_y, self.m_depth = self.clip(event.x, event.y, self.m_depth)
        self.s_x = self.m_x
        self.s_y = self.m_y
        self.s_depth = self.m_depth
        self.draw_stuff()

    def mouse1release(self, event):
        self.record = False
        self.draw_stuff()

    def mousescrolldown(self, event):
        # According to docs, I need to divide data by 120, idk why
        if self.record or mouse_scroll_force_update:
            self.m_depth -= 10
            self.m_depth = max(300, self.m_depth)
            self.s_depth = self.m_depth
            self.draw_stuff()
            print("Current Depth: {}".format(self.s_depth))

    def mousescrollup(self, event):
        # According to docs, I need to divide data by 120, idk why
        if self.record or mouse_scroll_force_update:
            self.m_depth += 10
            self.m_depth = min(10000, self.m_depth)
            self.s_depth = self.m_depth
            self.draw_stuff()
            print("Current Depth: {}".format(self.s_depth))
#            send_data_to_renderer(self.old_x_event, self.old_y_event, self.current_depth)

    def draw_stuff(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(
                canvas_border, canvas_border,
                canvas_width+canvas_border, canvas_height+canvas_border,
                fill="black")
        cirsize = math.sqrt(self.s_depth*2)/4
        self.canvas.create_oval(self.sent_x-cirsize, self.sent_y-cirsize, self.sent_x+cirsize, self.sent_y+cirsize,
                outline="white", width=2)
        self.canvas.create_oval(self.s_x-cirsize, self.s_y-cirsize, self.s_x+cirsize, self.s_y+cirsize,
                outline="red", width=2)

        self.canvas.create_oval(self.m_x-3, self.m_y-3, self.m_x+2, self.m_y+2,
                fill="green" if self.record else "red", outline="")
        self.canvas.pack(fill=BOTH, expand=1)

    def clip(self, x, y, depth):
        x = max(canvas_border, x)
        x = min(x, canvas_width+canvas_border)
        y = max(canvas_border, y)
        y = min(y, canvas_height+canvas_border)
        depth = max(300, depth)
        depth = min(10000, depth)
        return x, y, depth

    def interval(self):
        if self.s_x != self.sent_x or self.s_y != self.sent_y or self.s_depth != self.sent_depth:
            send_data_to_renderer(self.s_x, self.s_y, self.s_depth)
            self.sent_x =self.s_x
            self.sent_y =self.s_y
            self.sent_depth =self.s_depth
        self.canvas.after(time_interval, self.interval)


# Global Var that starts the position_server
position_server = PositionServer(4007)

def send_data_to_renderer(raw_x, raw_y, depth):
    """Converts raw mouse x and y data, to canonical coords -1 to 1"""
    x_no_border = raw_x - canvas_border
    y_no_border = raw_y - canvas_border
    render_data_x = x_no_border/canvas_width*2-1
    render_data_y = -(y_no_border/canvas_height*2-1)
    render_depth = depth;
    print("Raw Mouse: (%s %s)" % (raw_x, raw_y))
    print("No border Mouse: (%s %s)" % (x_no_border, y_no_border))
    print("Render data: (%s %s %s)" % (render_data_x, render_data_y, render_depth))
    position_server.set_values(render_data_x, render_data_y, render_depth)


if __name__ == "__main__":
    main()

