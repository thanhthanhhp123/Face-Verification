#using multi-threading and tkinter to create a GUI for the attendance system
import tkinter as tk
from tkinter import messagebox
from tkinter import *
from tkinter import ttk
from tkinter import filedialog

import threading
from threading import Thread
import time
import os
import cv2
import datetime
import csv

from PIL import Image, ImageTk
import numpy as np
import pickle
import torch
import tqdm

#Load the embeddings database
with open('database/embeddings_db.pkl', 'rb') as file:
    embeddings_db = pickle.load(file)

#Create the GUI
class AttendanceSystem:
    def __init__(self, window):
        self.window = window
        self.window.title('Attendance System')
        self.window.geometry('1000x600')
        self.window.configure(bg='white')
        self.window.resizable(False, False)
        self.window.iconbitmap('icons/attendance.ico')
        
        self.video_source = 1
        self.vid = cv2.VideoCapture(self.video_source)
        
        self.canvas = Canvas(window, width = 640, height = 480)
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        
        self.btn_snapshot = Button(window, text='Snapshot', width=20, command=self.snapshot)
        self.btn_snapshot.grid(row=1, column=0, padx=10, pady=10)
        
        self.btn_start = Button(window, text='Start', width=20, command=self.start)
        self.btn_start.grid(row=2, column=0, padx=10, pady=10)
        
        self.btn_stop = Button(window, text='Stop', width=20, command=self.stop)
        self.btn_stop.grid(row=3, column=0, padx=10, pady=10)
        
        self.btn_save = Button(window, text='Save', width=20, command=self.save)
        self.btn_save.grid(row=4, column=0, padx=10, pady=10)
        
        self.btn_quit = Button(window, text='Quit', width=20, command=self.quit)
        self.btn_quit.grid(row=5, column=0, padx=10, pady=10)
        
        self.lbl_name = Label(window, text='Name:', font=('Arial', 14), bg='white')
        self.lbl_name.grid(row=0, column=1, padx=10, pady=10)
        
        self.entry_name = Entry(window, font=('Arial', 14))
        self.entry_name.grid(row=0, column=2, padx=10, pady=10)
        
        self.lbl_status = Label(window, text='Status:', font=('Arial', 14), bg='white')
        self.lbl_status.grid(row=1, column=1, padx=10, pady=10)
        
        self.lbl_status_value = Label(window, text='Not Started', font=('Arial', 14), bg='white')
        self.lbl_status_value.grid(row=1, column=2, padx=10, pady=10)

        self.lbl_attendance = Label(window, text='Attendance:', font=('Arial', 14), bg='white')
        self.lbl_attendance.grid(row=2, column=1, padx=10, pady=10)

        self.lbl_attendance_value = Label(window, text='', font=('Arial', 14), bg='white')
        self.lbl_attendance_value.grid(row=2, column=2, padx=10, pady=10)

        self.lbl_time = Label(window, text='Time:', font=('Arial', 14), bg='white')
        self.lbl_time.grid(row=3, column=1, padx=10, pady=10)

        self.lbl_time_value = Label(window, text='', font=('Arial', 14), bg='white')
        self.lbl_time_value.grid(row=3, column=2, padx=10, pady=10)

        self.lbl_image = Label(window, text='Image:', font=('Arial', 14), bg='white')
        self.lbl_image.grid(row=4, column=1, padx=10, pady=10)

        self.canvas_image = Canvas(window, width = 200, height = 200)
        self.canvas_image.grid(row=4, column=2, padx=10, pady=10)

        self.update()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            cv2.imwrite('snapshots/snapshot.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            messagebox.showinfo('Snapshot', 'Snapshot saved successfully!')
        else:
            messagebox.showerror('Snapshot', 'Failed to save snapshot!')

    def start(self):
        self.lbl_status_value.config(text='Started')
        self.vid = cv2.VideoCapture(self.video_source)
        self.update()

    def stop(self):
        self.lbl_status_value.config(text='Stopped')
        self.vid.release()

    def save(self):
        name = self.entry_name.get()
        if name == '':
            messagebox.showerror('Save', 'Please enter a name!')
            return
        with open('attendance.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            now = datetime.datetime.now()
            writer.writerow([name, now.strftime("%Y-%m-%d %H:%M:%S")])
        self.entry_name.delete(0, 'end')
        self.update()

    def quit(self):
        self.window.destroy()

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(image=frame)
            self.canvas.create_image(0, 0, image=frame, anchor=tk.NW)
            self.canvas.image = frame
        self.window.after(10, self.update)
#Run
if __name__ == '__main__':
    window = tk.Tk()
    app = AttendanceSystem(window)
    window.mainloop()
