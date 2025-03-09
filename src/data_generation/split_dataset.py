import os
import random
import shutil

# Paths (relative to src/data_generation/)
current_path = os.getcwd()
img_dir = os.path.join(current_path, "..", "..", "data", "images")
label_dir = os.path.join(current_path, "..", "..", "data", "labels")
train_img_dir = os.path.join(current_path, "..", "..", "data", "train", "images")
train_label_dir = os.path.join(current_path, "..", "..", "data", "train", "labels")
val_img_dir = os.path.join(current_path, "..", "..", "data", "val", "images")
val_label_dir = os.path.join(current_path, "..", "..", "data", "val", "labels")
test_img_dir = os.path.join(current_path, "..", "..", "data", "test", "images")
test_label_dir = os.path.join(current_path, "..", "..", "data", "test", "labels")

# Create output directories
for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Get all image files
img_files = [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
random.shuffle(img_files)  # Randomize for even distribution

# Calculate split sizes (80% train, 10% val, 10% test)
total_files = len(img_files)
train_count = int(total_files * 0.8)
val_count = int(total_files * 0.1)
test_count = total_files - train_count - val_count  # Remainder to test

# Split files
train_files = img_files[:train_count]
val_files = img_files[train_count:train_count + val_count]
test_files = img_files[train_count + val_count:]

# Copy files to respective folders
for split, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
    for file in file_list:
        # Image and label paths
        img_src = os.path.join(img_dir, file)
        label_file = file.replace(".jpg", ".txt")
        label_src = os.path.join(label_dir, label_file)

        # Destination paths
        if split == "train":
            img_dst = os.path.join(train_img_dir, file)
            label_dst = os.path.join(train_label_dir, label_file)
        elif split == "val":
            img_dst = os.path.join(val_img_dir, file)
            label_dst = os.path.join(val_label_dir, label_file)
        else:  # test
            img_dst = os.path.join(test_img_dir, file)
            label_dst = os.path.join(test_label_dir, label_file)

        # Copy files
        shutil.copy2(img_src, img_dst)
        shutil.copy2(label_src, label_dst)
        print(f"Copied {file} and {label_file} to {split}")

print(f"Split complete: Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")