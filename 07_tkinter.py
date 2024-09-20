# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 12:52:14 2024

@author: lunel
"""

import tkinter as tk

def calculate():
    try:
        x = float(entry_x.get())
        y = float(entry_y.get())
        result = x + y
        label_result.config(text=f"{x} + {y}= {result}")
    except ValueError:
        label_result.config(text="put neumerics")

# make a window
root = tk.Tk()
root.title("x + y calculator")

# input x
label_x = tk.Label(root, text="x:")
label_x.pack(pady=5)
entry_x = tk.Entry(root)
entry_x.pack(pady=5)

# input y
label_y = tk.Label(root, text="y:")
label_y.pack(pady=5)
entry_y = tk.Entry(root)
entry_y.pack(pady=5)

# cal button
button = tk.Button(root, text="calculation", command=calculate)
button.pack(pady=10)

# result label
label_result = tk.Label(root, text="Result here:")
label_result.pack(pady=10)

# main loop
root.mainloop()
