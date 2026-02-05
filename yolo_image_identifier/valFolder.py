import os
import shutil
import random
import math

def copy_images_to_val(train_dir="mushroom_dataset/train", val_dir="mushroom_dataset/val", percent=0.2):
    """
    Copies images from each subfolder in `train_dir` to `val_dir`.
    If percent < 0.1, only one image is copied per class (if available).
    """
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    for folder_name in os.listdir(train_dir):
        train_folder = os.path.join(train_dir, folder_name)
        val_folder = os.path.join(val_dir, folder_name)

        if not os.path.isdir(train_folder):
            continue

        os.makedirs(val_folder, exist_ok=True)

        image_files = [
            f for f in os.listdir(train_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))
        ]

        if not image_files:
            continue

        # Decide number of images to copy
        if percent < 0.2:
            num_to_copy = 1
        else:
            num_to_copy = max(1, math.floor(len(image_files) * percent))

        images_to_copy = random.sample(image_files, min(num_to_copy, len(image_files)))

        for image in images_to_copy:
            src_path = os.path.join(train_folder, image)
            dst_path = os.path.join(val_folder, image)
            shutil.copy2(src_path, dst_path)

        print(f"Copied {len(images_to_copy)} images from '{folder_name}' to val set.")

# Run the function
copy_images_to_val()
