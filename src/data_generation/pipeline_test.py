import cv2
import numpy as np
import os

# Paths (your correct ones)
raw_img_dir = "../../data/raw/"
corner_label_dir = "../../data/roboflow_corner/labels/"
outline_label_dir = "../../data/roboflow_outline/labels/"
bg_dir = "backgrounds/"
out_img_dir = "../../data/synthetic/images/"
out_label_dir = "../../data/synthetic/labels/"
os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_label_dir, exist_ok=True)

# Backgrounds
bg_files = [f for f in os.listdir(bg_dir) if f.endswith(".jpg")]

# Single pass through deck
for img_file in os.listdir(raw_img_dir):
    print(f"Processing: {img_file}")
    
    if not (img_file.lower().endswith(".jpg")): 
        print(f"Skipped - not .jpg/JPG: {img_file}")
        continue
    
    # Load image and labels
    img_path = os.path.join(raw_img_dir, img_file)
    corner_label_path = os.path.join(corner_label_dir, img_file.lower().replace(".jpg", ".txt").replace(".JPG", ".txt"))
    outline_label_path = os.path.join(outline_label_dir, img_file.lower().replace(".jpg", ".txt").replace(".JPG", ".txt"))
    
    img = cv2.imread(img_path)
    if img is None: 
        print(f"Skipped - image load failed: {img_path}")
        continue
    
    try:
        with open(corner_label_path, "r") as f:
            corner_coords = list(map(float, f.read().split()))
            corner_class, corner_x, corner_y, corner_w, corner_h = corner_coords
    except FileNotFoundError:
        print(f"Skipped - corner label missing: {corner_label_path}")
        continue
    
    try:
        with open(outline_label_path, "r") as f:
            outline_coords = list(map(float, f.read().split()))
            outline_class, outline_x, outline_y, outline_w, outline_h = outline_coords
    except FileNotFoundError:
        print(f"Skipped - outline label missing: {outline_label_path}")
        continue
    
    h, w = img.shape[:2]

    # Blur, Contrast, Noise
    odd_sizes = [5, 7, 9, 11, 13, 15]
    blur_kernel = (np.random.choice(odd_sizes), np.random.choice(odd_sizes))
    noise_std = np.random.uniform(10, 30)
    contrast_alpha = np.random.uniform(0.8, 1.2)
    
    blurred = cv2.GaussianBlur(img, blur_kernel, 0)
    noise = np.random.normal(0, noise_std, blurred.shape).astype(np.uint8)
    noisy = cv2.add(blurred, noise)
    contrasted = cv2.convertScaleAbs(noisy, alpha=contrast_alpha)

    # Calculate new size for rotation (handle ±30° max, no clipping)
    theta = np.random.uniform(-30, 30)
    angle_rad = np.radians(abs(theta))
    new_w = int(w * np.cos(angle_rad) + h * np.sin(angle_rad))
    new_h = int(w * np.sin(angle_rad) + h * np.cos(angle_rad))
    center = (new_w // 2, new_h // 2)

    # Expand canvas (black background)
    expanded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    x_offset = (new_w - w) // 2
    y_offset = (new_h - h) // 2
    expanded[y_offset:y_offset+h, x_offset:x_offset+w] = contrasted

    # Rotation
    M = cv2.getRotationMatrix2D(center, theta, 1.0)
    # Adjust M for offset
    M[0, 2] += x_offset
    M[1, 2] += y_offset
    rotated_img = cv2.warpAffine(expanded, M, (new_w, new_h))

    # Rotate corner coords
    corner_x_px = corner_x * w + x_offset
    corner_y_px = corner_y * h + y_offset
    corner_center = np.array([corner_x_px, corner_y_px, 1])
    new_corner_center = M.dot(corner_center)
    new_corner_x = (new_corner_center[0] - x_offset) / w  # Normalize to original w/h
    new_corner_y = (new_corner_center[1] - y_offset) / h
    new_corner_label = f"{int(corner_class)} {new_corner_x} {new_corner_y} {corner_w} {corner_h}"

    # Mask with outline coords (before rotation, adjust for expanded size)
    outline_x_px = outline_x * w + x_offset
    outline_y_px = outline_y * h + y_offset
    orig_x_min = int(outline_x_px - (outline_w * w) / 2)
    orig_x_max = int(outline_x_px + (outline_w * w) / 2)
    orig_y_min = int(outline_y_px - (outline_h * h) / 2)
    orig_y_max = int(outline_y_px + (outline_h * h) / 2)
    mask = np.zeros((new_h, new_w), dtype=np.uint8)  # 2D mask
    mask[orig_y_min:orig_y_max, orig_x_min:orig_x_max] = 255
    # Convert mask to 3-channel for np.where
    rotated_mask = cv2.warpAffine(mask, M, (new_w, new_h))
    rotated_mask_3ch = cv2.cvtColor(rotated_mask, cv2.COLOR_GRAY2BGR)  # Make it 3-channel

    # Background swap
    bg_file = np.random.choice(bg_files)
    bg = cv2.resize(cv2.imread(os.path.join(bg_dir, bg_file)), (new_w, new_h))
    result = np.where(rotated_mask_3ch == 255, rotated_img, bg)

    # Save full image (no crop, keeps diversity—zoomed in/out)
    card_name = img_file[:-4]
    out_img = f"{card_name}_00.jpg"
    out_label = f"{card_name}_00.txt"
    cv2.imwrite(os.path.join(out_img_dir, out_img), result)
    with open(os.path.join(out_label_dir, out_label), "w") as f:
        f.write(new_corner_label)

print("Synthetic data generated")