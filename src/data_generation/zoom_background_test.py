import cv2
import numpy as np

# Load image and background
img = cv2.imread("7c.jpg")
bg = cv2.imread("bg.jpg")  # Replace with your bg file name
if img is None: exit("Image not found")
if bg is None: exit("Background not found")
h, w = img.shape[:2]  # 2061, 1940

# Original bounding box
class_id, x_cent, y_cent, box_w, box_h = 0, 0.49921875, 0.4859375, 0.57421875, 0.7265625

# Zoom factor (shrink)
zoom = 0.5
new_w, new_h = int(w * zoom), int(h * zoom)

# Shrink image
shrunk_img = cv2.resize(img, (new_w, new_h))

# Resize background
bg = cv2.resize(bg, (w, h))

# Create mask for shrunk image
mask = np.zeros((h, w), dtype=np.uint8)
x_offset = (w - new_w) // 2
y_offset = (h - new_h) // 2
mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = 255

# Blend with background
result = bg.copy()
result[mask == 255] = shrunk_img[mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w] == 255]

# New bounding box in pixels
new_box_w_px = int(box_w * w * zoom)
new_box_h_px = int(box_h * h * zoom)
new_x_cent_px = x_offset + (new_w // 2)
new_y_cent_px = y_offset + (new_h // 2)
x_min = max(0, new_x_cent_px - new_box_w_px // 2)
x_max = min(w, new_x_cent_px + new_box_w_px // 2)
y_min = max(0, new_y_cent_px - new_box_h_px // 2)
y_max = min(h, new_y_cent_px + new_box_h_px // 2)

# Crop around bounding box
cropped_result = result[y_min:y_max, x_min:x_max]

# Adjust bounding box for cropped image
crop_w, crop_h = x_max - x_min, y_max - y_min
new_x_cent = (new_x_cent_px - x_min) / crop_w
new_y_cent = (new_y_cent_px - y_min) / crop_h
new_box_w = new_box_w_px / crop_w
new_box_h = new_box_h_px / crop_h
new_label = f"{class_id} {new_x_cent} {new_y_cent} {new_box_w} {new_box_h}"

# Save and print
cv2.imwrite("zoom_bg_test.jpg", cropped_result)
print(f"Zoom factor: {zoom}")
print(f"New coords: {new_label}")
print(f"Cropped size: {cropped_result.shape[1]}x{cropped_result.shape[0]}")