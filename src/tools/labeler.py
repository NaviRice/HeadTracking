import tkinter as tk
from tkinter import messagebox as tkMessageBox
from PIL import Image, ImageTk
import numpy as np

from get_one_img import get_depth_and_ir_from_kinect
from enum import Enum
import pickle

import cv2

class Selection(Enum):
    NOTHING = 1
    FACE = 2
    LEFT_EYE = 3
    RIGHT_EYE = 4

class SelectEyesDialog(tk.Toplevel):
    def __init__(self, parent):
        
        tk.Toplevel.__init__(self, parent)
        self.transient(parent)

        self.title("Select Eyes")
        
        self._app = parent

        rect = self._app._face_rect

        self._ir_image_data  = self._app._ir_image_data[rect[1]:rect[3], rect[0]:rect[2]]

        canvas_width = rect[3] - rect[1]
        canvas_height  = rect[2] - rect[0]

        ir_image=Image.fromarray(self._ir_image_data)

        self._canvas_frame = tk.Frame(self)
        self._canvas_depth = tk.Canvas(self._canvas_frame, 
           width=canvas_width,
           height=canvas_height)

        self._canvas_ir = tk.Canvas(self._canvas_frame, 
           width=canvas_width,
           height=canvas_height)
        
        self._canvas_frame.grid(row=0)
        self._canvas_ir.grid(row=0, column=0)
        self._canvas_depth.grid(row=0, column=1)


        self._canvas_ir.delete("all")

        self._canvas_ir.ir_photo = ImageTk.PhotoImage(ir_image)
        self._canvas_ir.create_image(0, 0, image=self._canvas_ir.ir_photo, anchor=tk.NW)

        self._done_btn = tk.Button(self, text="Done", command=self.ok)
        self._done_btn.grid(row=1)

        self.grab_set()

    def ok(self):
        self.top.destroy()

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()

        self._start_x = 0
        self._start_y = 0
        self._end_x = 0
        self._end_y = 0

        self._count = 0

        self._face_rect = None
        self._left_eye_rect = None
        self._right_eye_rect = None

        self._face_selected = False
        self._left_eye_selected = False
        self._right_eye_selected = False

        self._create_widgets()
        self._selection = Selection.NOTHING

        self._depth_images = []
        self._ir_images = []
        self._labels = []

    def _normalize(self, image_data):
        ir_max = np.max(image_data)
        image_data = image_data / ir_max * 255

        return np.reshape(image_data, (image_data.shape[0], image_data.shape[1]))

    def _create_widgets(self):
        canvas_width = 512
        canvas_height = 424

        self._hot_key_labels = tk.Label(self, text="W = Select Face(Blue)    E = Select Left Eye(Green)    R = Select Right Eye(Red)     Space = Next Image")


        self._canvas_frame = tk.Frame(self)
        self._canvas_depth = tk.Canvas(self._canvas_frame, 
           width=canvas_width,
           height=canvas_height)

        self._canvas_ir = tk.Canvas(self._canvas_frame, 
           width=canvas_width,
           height=canvas_height)

        self._button_frame = tk.Frame(self)

        self._select_face_btn = tk.Button(self._button_frame, text="Select Face", command=self._select_face)
        self._select_left_eye_btn = tk.Button(self._button_frame, text="Select Left Eye", command=self._select_left_eye)
        self._select_right_eye_btn = tk.Button(self._button_frame, text="Select Right Eye", command=self._select_right_eye)
        self._next_btn = tk.Button(self._button_frame, text="Next", command=self._next)

        self._hot_key_labels.grid(row=0)

        self._canvas_frame.grid(row=1)
        self._canvas_ir.grid(row=0, column=0)
        self._canvas_depth.grid(row=0, column=1)

        self._button_frame.grid(row=2, column=0)
        self._select_face_btn.grid(row=0, column=0)
        self._select_left_eye_btn.grid(row=0, column=1)
        self._select_right_eye_btn.grid(row=0, column=2)
        self._next_btn.grid(row=0, column=3)

        self._init_new_images()

        self._canvas_ir.bind("<Button-1>", self._start_selection)
        self._canvas_ir.bind("<B1-Motion>", self._show_selection)
        self._canvas_ir.bind("<ButtonRelease-1>", self._update_selection)

        self.master.bind("<Key>", self._key)
    
    def _key(self, event):
        key = repr(event.char)

        if key == "'w'":
            self._select_face()

        elif key == "'e'":
            self._select_left_eye()

        elif key == "'r'":
            self._select_right_eye()

        elif key == "'0'":
            self._face_zoom_in()

        elif key == "' '":
            self._next()

    def _face_zoom_in(self):
        if not self._face_selected:
            return

        selectEyesDialog = SelectEyesDialog(self)
        self.master.wait_window(selectEyesDialog)

    def _select_face(self):
        self._selection = Selection.FACE
        self._update_title()
    
    def _select_left_eye(self):
        self._selection = Selection.LEFT_EYE
        self._update_title()
    
    def _select_right_eye(self):
        self._selection = Selection.RIGHT_EYE
        self._update_title()

    def _right(self, event):
        self._next()

    def _init_new_images(self):
        self._selection = Selection.NOTHING
        self._face_selected = False
        self._left_eye_selected = False
        self._right_eye_selected = False

        self._depth_image_data, self._ir_image_data = get_depth_and_ir_from_kinect()

        self._ir_image_data = self._normalize(self._ir_image_data)
        self._depth_image_data = self._normalize(self._depth_image_data)
        
        ir_image=Image.fromarray(self._ir_image_data)
        depth_image=Image.fromarray(self._depth_image_data)

        self._canvas_ir.ir_photo = ImageTk.PhotoImage(ir_image)
        self._canvas_depth.depth_photo = ImageTk.PhotoImage(depth_image)

        self._canvas_ir.delete("all")
        self._canvas_depth.delete("all")

        self._canvas_ir.create_image(0, 0, image=self._canvas_ir.ir_photo, anchor=tk.NW)
        self._canvas_depth.create_image(0, 0, image=self._canvas_depth.depth_photo, anchor=tk.NW)
        self._update_title()

    def _next(self):
        if not self._face_selected or not self._left_eye_selected or not self._right_eye_selected:
            tkMessageBox.showwarning(
                "Not Enough Label",
                "Please label face, left eye and right eye"
            )
            return

        self._count += 1

        self._depth_images.append(self._depth_image_data)
        self._ir_images.append(self._ir_image_data)
        self._labels.append({
            "face_rect": self._face_rect,
            "left_eye_rect": self._left_eye_rect,
            "right_eye_rect": self._right_eye_rect
        })

        with open('data/depth_images.pkl', 'wb') as output:
            pickle.dump(self._depth_images, output, pickle.HIGHEST_PROTOCOL)

        with open('data/ir_images.pkl', 'wb') as output:
            pickle.dump(self._ir_images, output, pickle.HIGHEST_PROTOCOL)
        
        with open('data/labels.pkl', 'wb') as output:
            pickle.dump(self._labels, output, pickle.HIGHEST_PROTOCOL)

        self._init_new_images()

    def _start_selection(self, event):
        if self._selection is Selection.NOTHING:
            tkMessageBox.showwarning(
                "No Feature To Label",
                "Please select a facial feature to label"
            )
            return

        self._start_x = event.x
        self._start_y = event.y
    
    def _show_selection(self, event):
        if self._selection is Selection.NOTHING:
            return

        self._end_x = event.x
        self._end_y = event.y

        self._canvas_ir.delete("all")
        self._canvas_depth.delete("all")

        self._canvas_ir.create_image(0, 0, image=self._canvas_ir.ir_photo, anchor=tk.NW)
        self._canvas_depth.create_image(0, 0, image=self._canvas_depth.depth_photo, anchor=tk.NW)

        color = ""
        if self._selection is Selection.FACE:
            color = "blue"
            if self._left_eye_selected:
                self._canvas_ir.create_rectangle(self._left_eye_rect[0], self._left_eye_rect[1], self._left_eye_rect[2], self._left_eye_rect[3], outline="green")
                self._canvas_depth.create_rectangle(self._left_eye_rect[0], self._left_eye_rect[1], self._left_eye_rect[2], self._left_eye_rect[3], outline="green")
            
            if self._right_eye_selected:
                self._canvas_ir.create_rectangle(self._right_eye_rect[0], self._right_eye_rect[1], self._right_eye_rect[2], self._right_eye_rect[3], outline="red")
                self._canvas_depth.create_rectangle(self._right_eye_rect[0], self._right_eye_rect[1], self._right_eye_rect[2], self._right_eye_rect[3], outline="red")

        elif self._selection is Selection.LEFT_EYE:
            color = "green"
            if self._face_selected:
                self._canvas_ir.create_rectangle(self._face_rect[0], self._face_rect[1], self._face_rect[2], self._face_rect[3], outline="blue")
                self._canvas_depth.create_rectangle(self._face_rect[0], self._face_rect[1], self._face_rect[2], self._face_rect[3], outline="blue")
            
            if self._right_eye_selected:
                self._canvas_ir.create_rectangle(self._right_eye_rect[0], self._right_eye_rect[1], self._right_eye_rect[2], self._right_eye_rect[3], outline="red")
                self._canvas_depth.create_rectangle(self._right_eye_rect[0], self._right_eye_rect[1], self._right_eye_rect[2], self._right_eye_rect[3], outline="red")
        
        elif self._selection is Selection.RIGHT_EYE:
            color = "red"
            
            if self._face_selected:
                self._canvas_ir.create_rectangle(self._face_rect[0], self._face_rect[1], self._face_rect[2], self._face_rect[3], outline="blue")
                self._canvas_depth.create_rectangle(self._face_rect[0], self._face_rect[1], self._face_rect[2], self._face_rect[3], outline="blue")
            
            if self._left_eye_selected:
                self._canvas_ir.create_rectangle(self._left_eye_rect[0], self._left_eye_rect[1], self._left_eye_rect[2], self._left_eye_rect[3], outline="green")
                self._canvas_depth.create_rectangle(self._left_eye_rect[0], self._left_eye_rect[1], self._left_eye_rect[2], self._left_eye_rect[3], outline="green")

        self._canvas_ir.create_rectangle(self._start_x, self._start_y, self._end_x, self._end_y, outline=color)
        self._canvas_depth.create_rectangle(self._start_x, self._start_y, self._end_x, self._end_y, outline=color)

    def _update_selection(self, event):
        if self._selection is Selection.LEFT_EYE:
            self._left_eye_rect = (self._start_x, self._start_y, self._end_x, self._end_y)
            self._left_eye_selected = True
        
        elif self._selection is Selection.RIGHT_EYE:
            self._right_eye_rect = (self._start_x, self._start_y, self._end_x, self._end_y)
            self._right_eye_selected = True

        elif self._selection is Selection.FACE:
            self._face_rect = (self._start_x, self._start_y, self._end_x, self._end_y)
            self._face_selected = True
        
        self._update_title()

    def _update_title(self):
        face_selected = ""
        if self._face_selected:
            face_selected = "FACE"
        
        left_eye_selected = ""
        if self._left_eye_selected:
            left_eye_selected = "LEFT_EYE"

        right_eye_selected = ""
        if self._right_eye_selected:
            right_eye_selected = "RIGHT_EYE"
        
        selections = "%s %s %s" % (face_selected, left_eye_selected, right_eye_selected)

        if selections != "  ":
            selections = "%s -" % selections

        selecting = ""
        if self._selection is Selection.LEFT_EYE:
            selecting = "Selecting Left Eye -"
        if self._selection is Selection.RIGHT_EYE:
            selecting = "Selecting Right Eye -"
        elif self._selection is Selection.FACE:
            selecting = "Selecting Face -"
        
        self.master.title("%s %s Labled %d samples" % (selections, selecting, self._count))

def main():
    root = tk.Tk()
    app = Application(master=root)

    root.lift()
    app.mainloop()

if __name__ == "__main__":
    main()