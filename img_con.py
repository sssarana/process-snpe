import numpy as np
import cv2

# Load and resize the image
img = cv2.imread('/home/yolov5/person.jpg')
img_resized = cv2.resize(img, (640, 640))

# Convert to float32 and normalize
img_float = img_resized.astype(np.float32)
img_normalized = img_float / 255.0

# Save as raw binary file
img_normalized.tofile('person.raw')
