import cv2

# Load image
img = cv2.imread("max_test3.jpg")
if img is None: exit("Image not found")

# Adjust contrast (alpha > 1 increases, < 1 decreases)
alpha = 1.6  # generate random value from .5 to 2.5 
contrast_img = cv2.convertScaleAbs(img, alpha=alpha)

# Save
cv2.imwrite("max_test3.jpg", contrast_img)

