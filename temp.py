# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Hide the root tkinter window
Tk().withdraw()

# Open file selection dialog
file_path = askopenfilename(title="Select an image file", filetypes=[("TIFF files", "*.tif;*.tiff"), ("All files", "*.*")])

# Check if a file was selected
if file_path:
    # Open the TIFF image using Pillow
    image = Image.open(file_path)


# rotate the image
rotated_image = image.rotate(90, expand=True)


# Convert the image to a NumPy array
image_array = np.array(rotated_image)


# make a binary_image
threshold = 40
binary_image_array = (image_array > threshold) * 255


max_value = np.max(binary_image_array)
print("MAX:", max_value)

# change values in the image
binary_image_array[100:300, 100:300] = 200
print(binary_image_array)


# Display the binary image
plt.imshow(binary_image_array, cmap='gray')
# let's ask what kind of colors imshow have.


# image setting
plt.axis('off')
plt.title('Thresholded Binary Image')


#show the image on a display
plt.show()

