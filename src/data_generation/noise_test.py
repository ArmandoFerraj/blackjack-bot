import cv2
import numpy as np

# Load image
img = cv2.imread("7c.jpg")
if img is None: exit("Image not found")

# Generate random noise (same shape)
noise = np.random.normal(0, 5, img.shape).astype(np.uint8)  # Mean 0, std 25. std should random value between 0 - 5

# Add noise to image
noisy_img = cv2.add(img, noise)

# Save
cv2.imwrite("max_test3.jpg", noisy_img)