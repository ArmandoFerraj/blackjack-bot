import cv2
import numpy as np

# Load image
img = cv2.imread("7c.jpg")
if img is None: exit("Image not found")
h, w = img.shape[:2]  # 2061, 1940

# Original bounding box
class_id, x_cent, y_cent, box_w, box_h = 0, 0.49921875, 0.4859375, 0.57421875, 0.7265625

# Random rotation angle
theta = np.random.uniform(-30, 30)
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, theta, 1.0)

# Rotate image
rotated_img = cv2.warpAffine(img, M, (w, h))

# Rotate bounding box center
x_cent_px = x_cent * w
y_cent_px = y_cent * h
center_point = np.array([x_cent_px, y_cent_px, 1])
new_center = M.dot(center_point)
new_x_cent = new_center[0] / w
new_y_cent = new_center[1] / h

# New bounding box
new_label = f"{class_id} {new_x_cent} {new_y_cent} {box_w} {box_h}"

# Save and print
cv2.imwrite("rotated_test.jpg", rotated_img)
print(f"Theta: {theta}")
print(f"New coords: {new_label}")