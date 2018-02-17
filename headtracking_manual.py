from threading import Thread
from tkinter import *


# Global vars that change the way canvas looks
canvas_border = 50
canvas_width = canvas_height = 500

class Window(Frame):

    def __init__(self, master=None):
        """Initializes starting vars.

        data_x = the x position of the mouse from -1 to 1 relative to canvas.
        data_y = the y position of the mouse from -1 to 1 relative to canvas.
        data_z = the depth value changed by the scrollwheel of mouse
        """
        master.minsize(
                width=canvas_width + 2*canvas_border,
                height=canvas_height + 2*canvas_border)
        Frame.__init__(self, master)
        self.master = master
        self.init_window()
        self.mouse_down = False;
        self.data_x = 0;
        self.data_y = 0;
        self.data_z = 300; # Kinects closest depth is 300
        self.record = False

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

    def mousemove(self, event):
        circle_color = "red"
        x, y = self.clip(event.x, event.y)
        if self.record is True:
            send_data_to_renderer(x, y)
            circle_color = "green"
        self.draw_circle(event.x, event.y, circle_color)

    def mouse1press(self, event):
        self.record = True
        self.draw_circle(event.x, event.y, "green")
        send_data_to_renderer(event.x, event.y)

    def mouse1release(self, event):
        self.record = False
        self.draw_circle(event.x, event.y, "red")

    def draw_circle(self, x, y, color):
        x, y = self.clip(x, y)
        self.canvas.delete("all")
        self.canvas.create_rectangle(
                canvas_border, canvas_border,
                canvas_width+canvas_border, canvas_height+canvas_border,
                fill="#476042")
        self.canvas.create_oval(x-5, y-5, x+5, y+5,
                outline="black", fill=color, width=2)
        self.canvas.pack(fill=BOTH, expand=1)

    def clip(self, x, y):
        x = max(canvas_border, x)
        x = min(x, canvas_width+canvas_border)
        y = max(canvas_border, y)
        y = min(y, canvas_height+canvas_border)
        return x, y

def send_data_to_renderer(raw_x, raw_y):
    """Converts raw mouse x and y data, to canonical coords -1 to 1"""
    x_no_border = raw_x - canvas_border
    y_no_border = raw_y - canvas_border
    render_data_x = x_no_border/canvas_width*2-1
    render_data_y = y_no_border/canvas_height*2-1
    print("Raw Mouse: (%s %s)" % (raw_x, raw_y))
    print("No border Mouse: (%s %s)" % (x_no_border, y_no_border))
    print("Render data: (%s %s)" % (render_data_x, render_data_y))
    return

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

if __name__ == "__main__":
    main()

