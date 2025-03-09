import os
import random
import shutil

# Current path (src/data_generation/)
current_path = os.getcwd()

# Paths
synth_img_dir = os.path.join(current_path, "..", "..", "data", "synthetic", "images")
synth_label_dir = os.path.join(current_path, "..", "..", "data", "synthetic", "labels")
train_img_dir = os.path.join(current_path, "..", "..", "data", "train", "images")
train_label_dir = os.path.join(current_path, "..", "..", "data", "train", "labels")
val_img_dir = os.path.join(current_path, "..", "..", "data", "val", "images")
val_label_dir = os.path.join(current_path, "..", "..", "data", "val", "labels")
test_img_dir = os.path.join(current_path, "..", "..", "data", "test", "images")
test_label_dir = os.path.join(current_path, "..", "..", "data", "test", "labels")

# Create output directories
for dir_path in [train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Group files by card (e.g., 2c, 3d, ..., ah)
card_files = {}  # {card_name: [list of files]}
for img_file in os.listdir(synth_img_dir):
    if img_file.endswith(".jpg"):
        card_name = img_file[:-7]  # Strip "_XX.jpg" (e.g., "2c_00.jpg" -> "2c")
        if card_name not in card_files:
            card_files[card_name] = []
        card_files[card_name].append(img_file)

# Split each card's files (20 per card: 16 train, 2 val, 2 test)
for card_name, files in card_files.items():
    print(f"Processing card {card_name} ({len(files)} files)")
    random.shuffle(files)  # Randomize for diversity
    
    # Split: 16 train, 2 val, 2 test (80%/10%/10% of 20 = 16/2/2)
    train_files = files[:16]
    val_files = files[16:18]
    test_files = files[18:]
    
    # Copy files, maintaining image/label pairs
    for split, file_list in [("train", train_files), ("val", val_files), ("test", test_files)]:
        for file in file_list:
            img_src = os.path.join(synth_img_dir, file)
            label_file = file.replace(".jpg", ".txt")
            label_src = os.path.join(synth_label_dir, label_file)
            
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

print(f"Split complete: Train={16*len(card_files)}, Val={2*len(card_files)}, Test={2*len(card_files)}")