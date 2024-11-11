import os
import shutil
import json
import random
from pathlib import Path

# Define paths
input_dir = Path("C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-class-cropped/Train/images")
output_dir = Path("C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-class-cropped-organized")
labels_file = Path("C:/Users/Studio/Documents/GitHub/dice-reader/data/dice-class-cropped/Train/labels.json")

# Ensure output directory is clean
if output_dir.exists():
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Create subdirectories for train, test, and val
split_dirs = {
    "train": output_dir / "train",
    "test": output_dir / "test",
    "val": output_dir / "val"
}
for split_dir in split_dirs.values():
    split_dir.mkdir(parents=True, exist_ok=True)

# Load labels
with open(labels_file, 'r') as f:
    labels_data = json.load(f)

# Gather all image paths and create a lookup dictionary with and without .jpg
all_image_paths = list(input_dir.rglob("*.jpg"))  # Adjust for other formats if needed
image_path_dict = {}
for img_path in all_image_paths:
    image_path_dict[img_path.stem] = img_path  # without .jpg
    image_path_dict[img_path.name] = img_path  # with .jpg

# Track missing files
missing_files = []

# Organize images by class and split into train, test, and val
images_by_class = {}
for image_name, info in labels_data.items():
    label = info['labels'][0]
    if label not in images_by_class:
        images_by_class[label] = []
    images_by_class[label].append(image_name)

# Split data and copy images
for label, images in images_by_class.items():
    random.shuffle(images)
    train_count = int(len(images) * 0.4)
    test_count = int(len(images) * 0.4)

    for i, image_name in enumerate(images):
        # Determine split
        if i < train_count:
            split = "train"
        elif i < train_count + test_count:
            split = "test"
        else:
            split = "val"

        # Define destination path
        dest_dir = split_dirs[split] / label
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Remove directory prefix for lookup in image_path_dict
        stripped_image_name = os.path.basename(image_name)
        
        # Check if image exists in the gathered paths, with and without .jpg
        print(f"Attempting to locate {stripped_image_name} in image_path_dict...")
        source_path = image_path_dict.get(stripped_image_name) or image_path_dict.get(stripped_image_name + ".jpg")

        if source_path:
            dest_path = dest_dir / source_path.name
            shutil.copy(source_path, dest_path)
            print(f"Copied {source_path} to {dest_path}")
        else:
            print(f"Warning: File {stripped_image_name} not found.")
            missing_files.append(stripped_image_name)

# Report missing files if any
if missing_files:
    print("\nSome files were not found and were skipped:")
    for file in missing_files:
        print(file)
else:
    print("All files were organized successfully.")
