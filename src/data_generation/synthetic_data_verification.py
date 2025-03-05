import cv2
import numpy as np
import os

# Paths
synth_img_dir = "../../data/synthetic/images/"
synth_label_dir = "../../data/synthetic/labels/"
verify_dir = "../../data/synthetic/verification/"
os.makedirs(verify_dir, exist_ok=True)

# Process each synthetic image
for img_file in os.listdir(synth_img_dir):
    if not img_file.endswith(".jpg"): continue
    
    # Load image
    img_path = os.path.join(synth_img_dir, img_file)
    img = cv2.imread(img_path)
    if img is None: 
        print(f"Skipped - image load failed: {img_path}")
        continue
    
    # Load label (corner coords)
    label_path = os.path.join(synth_label_dir, img_file.replace(".jpg", ".txt"))
    try:
        with open(label_path, "r") as f:
            class_id, x_cent, y_cent, box_w, box_h = map(float, f.read().split())
    except FileNotFoundError:
        print(f"Skipped - label missing: {label_path}")
        continue
    
    # Get image dimensions
    h, w = img.shape[:2]

    # Convert normalized coords to pixels
    x_min = int((x_cent - box_w / 2) * w)
    x_max = int((x_cent + box_w / 2) * w)
    y_min = int((y_cent - box_h / 2) * h)
    y_max = int((y_cent + box_h / 2) * h)

    # Draw green lines connecting corners (rectangle)
    cv2.line(img, (x_min, y_min), (x_max, y_min), (0, 255, 0), 2)  # Top
    cv2.line(img, (x_max, y_min), (x_max, y_max), (0, 255, 0), 2)  # Right
    cv2.line(img, (x_max, y_max), (x_min, y_max), (0, 255, 0), 2)  # Bottom
    cv2.line(img, (x_min, y_max), (x_min, y_min), (0, 255, 0), 2)  # Left

    # Save to verification folder
    out_file = os.path.join(verify_dir, img_file)
    cv2.imwrite(out_file, img)
    print(f"Verified: {img_file}")

print("Verification complete")