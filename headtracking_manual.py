from position_server import PositionServer

from threading import Thread
from tkinter import *


# Global vars that change the way canvas looks
canvas_border = 50
canvas_width = canvas_height = 500

def main():
    root = Tk()
    root.resizable(width=False, height=False)
    root.geometry("170x65")
    root.attributes('-type', 'dialog')
    app = Window(root)
    def on_quit():
        app.kill()
        exit()
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
        self.mouse_down = False
        self.record = False
        self.current_x = 0
        self.current_y = 0
        self.current_depth = 300


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
                canvas_width+canvas_border, canvas_height+canvas_border, fill="#476042")
        self.canvas.bind("<Motion>", self.mousemove)
        self.canvas.bind("<ButtonPress-1>", self.mouse1press)
        self.canvas.bind("<ButtonRelease-1>", self.mouse1release)
        self.canvas.bind("<Button-4>", self.mousescrollup)
        self.canvas.bind("<Button-5>", self.mousescrolldown)

    def mousemove(self, event):
        circle_color = "red"
        x, y, depth = self.clip(event.x, event.y, self.current_depth)
        if self.record is True:
            send_data_to_renderer(x, y, self.current_depth)
            circle_color = "green"
        self.draw_circle(event.x, event.y, circle_color)

    def mouse1press(self, event):
        self.record = True
        self.draw_circle(event.x, event.y, "green")
        send_data_to_renderer(event.x, event.y, self.current_depth)

    def mouse1release(self, event):
        self.record = False
        self.draw_circle(event.x, event.y, "red")

    def mousescrolldown(self, event):
        # According to docs, I need to divide data by 120, idk why
        if self.record:
            self.current_depth -= 1
            self.current_depth = max(300, self.current_depth)
            print("Current Depth: {}".format(self.current_depth))
            send_data_to_renderer(
                self.current_x, self.current_y, self.current_depth)

    def mousescrollup(self, event):
        # According to docs, I need to divide data by 120, idk why
        if self.record:
            self.current_depth += 1
            print("Current Depth: {}".format(self.current_depth))
            send_data_to_renderer(
                self.current_x, self.current_y, self.current_depth)

    def draw_circle(self, x, y, color):
        x, y, depth = self.clip(x, y, self.current_depth)
        self.canvas.delete("all")
        self.canvas.create_rectangle(
                canvas_border, canvas_border,
                canvas_width+canvas_border, canvas_height+canvas_border,
                fill="#476042")
        self.canvas.create_oval(x-5, y-5, x+5, y+5,
                outline="black", fill=color, width=2)
        self.canvas.pack(fill=BOTH, expand=1)

    def clip(self, x, y, depth):
        x = max(canvas_border, x)
        x = min(x, canvas_width+canvas_border)
        y = max(canvas_border, y)
        y = min(y, canvas_height+canvas_border)
        # This depth capping also happens on mousescrolldown
        depth = max(300, depth)
        return x, y, depth

# Global Var that starts the position_server
position_server = PositionServer(40007)

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

