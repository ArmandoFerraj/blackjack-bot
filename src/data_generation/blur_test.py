import cv2

# Load image
img = cv2.imread("max_test3.jpg")
if img is None: exit("Image not found")

# Apply strong blur
blurred = cv2.GaussianBlur(img, (51, 51), 0) # generate random ODD values from 1 - 51

# Save
cv2.imwrite("max_test3.jpg", blurred)