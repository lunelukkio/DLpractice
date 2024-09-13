import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog

# for GUI
root = tk.Tk()
root.withdraw()  # Hide the root window

# open dialog
image_path = filedialog.askopenfilename(filetypes=[("TIFF files", "*.tif;*.tiff")])

# Open the TIFF image using Pillow and convert it to a NumPy array
img = Image.open(image_path)
img_array = np.array(img)

# rotate the image
rotated_array = np.rot90(img_array, k=1, axes=(0, 1))

# Display the image using matplotlib
plt.imshow(rotated_array, cmap='gray')
plt.title("TIFF Image")
plt.axis('off')  # Hide axis
plt.show()

# Optionally, you can print image details
print(f"Image shape: {img_array.shape}")
print(f"Image data type: {img_array.dtype}")
