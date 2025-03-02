import cv2
import numpy as np

# Load image
img = cv2.imread("7c.jpg")
if img is None: exit("Image not found")
h, w = img.shape[:2]  # 2061, 1940

# Original bounding box
class_id, x_cent, y_cent, box_w, box_h = 0, 0.49921875, 0.4859375, 0.57421875, 0.7265625

# Zoom factor (shrink: 0.5, no enlarge here)
zoom = 0.5  # Test with 0.5
new_w, new_h = int(w * zoom), int(h * zoom)

# Resize (shrink) image
shrunk_img = cv2.resize(img, (new_w, new_h))

# Create blank canvas (original size) and place shrunk image in center
canvas = np.zeros((h, w, 3), dtype=np.uint8)  # Black background
x_offset = (w - new_w) // 2  # Center horizontally
y_offset = (h - new_h) // 2  # Center vertically
canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = shrunk_img

# Adjust bounding box (center moves, size scales)
new_x_cent = (x_cent * w - x_offset) / w  # Adjust for new position
new_y_cent = (y_cent * h - y_offset) / h
new_box_w = box_w * zoom
new_box_h = box_h * zoom
new_label = f"{class_id} {new_x_cent} {new_y_cent} {new_box_w} {new_box_h}"

# Save and print
cv2.imwrite("zoomed_test.jpg", canvas)
print(f"Zoom factor: {zoom}")
print(f"New coords: {new_label}")
print(f"Canvas size: {canvas.shape[1]}x{canvas.shape[0]}")