import cv2
import numpy as np
import os

# Inputs from rotation_test.py
theta = -12.965974025380383  # Replace with your theta from rotation_test.py
class_id, x_cent, y_cent, box_w, box_h = 0, 0.5025328733540663, 0.48612485959390417, 0.57421875, 0.7265625

# Load rotated image
img = cv2.imread("rotated_test.jpg")
if img is None: exit("Image not found")
h, w = img.shape[:2]  # 2061, 1940

# Create mask with ORIGINAL coords (before rotation)
orig_x_min = int((0.49921875 - 0.57421875 / 2) * w)
orig_x_max = int((0.49921875 + 0.57421875 / 2) * w)
orig_y_min = int((0.4859375 - 0.7265625 / 2) * h)
orig_y_max = int((0.4859375 + 0.7265625 / 2) * h)
mask = np.zeros_like(img, dtype=np.uint8)
mask[orig_y_min:orig_y_max, orig_x_min:orig_x_max] = 255

# Rotate mask with theta
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, theta, 1.0)
rotated_mask = cv2.warpAffine(mask, M, (w, h))

# Background swap
background_dir = r"C:\Users\aferr\Desktop\bj_bot\src\data_generation\backgrounds"
for idx, file in enumerate(os.listdir(background_dir)):
    if file.endswith(".jpg"):
        bg_path = os.path.join(background_dir, file)
        bg = cv2.resize(cv2.imread(bg_path), (w, h))
        result = np.where(rotated_mask == 255, img, bg)
        cv2.imwrite(f"new_card_{idx}.jpg", result)

print("Done")