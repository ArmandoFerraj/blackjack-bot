import cv2
import numpy as np
import os

# Load image
img = cv2.imread("rotated_test.jpg")
h, w = img.shape[:2]  # 2061, 1940

# Coordinates
#x_cent, y_cent, box_w, box_h = 0.49921875, 0.4859375, 0.57421875, 0.7265625 # original bounding box coords

x_cent, y_cent, box_w, box_h = 0.4946919667973364, 0.48685133975832184, 0.57421875, 0.7265625 #new bounding box coords



# Convert to pixels
x_min = int((x_cent - box_w / 2) * w)  # 411
x_max = int((x_cent + box_w / 2) * w)  # 1525
y_min = int((y_cent - box_h / 2) * h)  # 253
y_max = int((y_cent + box_h / 2) * h)  # 1750

# Create mask
mask = np.zeros_like(img, dtype=np.uint8)  # Same shape as img, all black
mask[y_min:y_max, x_min:x_max] = 255



background_dir = r"C:\Users\aferr\Desktop\bj_bot\src\data_generation\backgrounds"

# Iterate through .jpg files
for idx, file in enumerate(os.listdir(background_dir)):
    if file.endswith(".jpg"):  # Filter for .jpg files
        bg_path = os.path.join(background_dir, file)  # Full path
        bg = cv2.imread(bg_path)
        bg = cv2.resize(bg, (1940, 2061))  # Resize to match img
        result = np.where(mask == 255, img, bg)
        
        # Save with unique name (e.g., new_card_0.jpg, new_card_1.jpg)
        cv2.imwrite(f"new_card_{idx}.jpg", result)



print("done")