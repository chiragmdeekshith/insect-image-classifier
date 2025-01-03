import os
import shutil
from pathlib import Path

# Define input and output paths
input_parent = "../Datasets/ImagesCV/"
categories = [
    "cockroach", "fly", "grasshopper", "isopod", "lacewing",
    "ladybug", "leafHopper", "long_horned_beetle", "mantis", "tigerBeetle"
]
output_parent = "../organized_datasets/ImagesCV/"

# Create output directories for each category
output_dirs = {
    "train": {category: os.path.join(output_parent, "training_data", category) for category in categories},
    "val": {category: os.path.join(output_parent, "validation_data", category) for category in categories},
    "test": {category: os.path.join(output_parent, "testing_data", category) for category in categories}
}

# Create the output directories if they don't exist
for dataset_type in output_dirs.values():
    for path in dataset_type.values():
        os.makedirs(path, exist_ok=True)

# Function to organize images based on file size
def organize_images(category, input_folder, output_folders):
    all_images = []

    # Step 1: Recursively gather images from `data/train`, `data/val`, and `data/test` subfolders
    subfolders = ["data/train", "data/val", "data/test"]
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for subfolder in subfolders:
        folder_path = Path(input_folder) / subfolder
        if folder_path.exists():
            # Recursively find all image files within the subfolder
            for img_path in folder_path.rglob("*"):
                if img_path.is_file() and img_path.suffix.lower() in valid_extensions:
                    all_images.append(img_path)
        else:
            print(f"Warning: Subfolder {folder_path} does not exist.")

    print(f"Found {len(all_images)} images for category '{category}'.")

    if not all_images:
        print(f"No images found for category '{category}'. Skipping...")
        return

    # Step 2: Sort images by descending size
    all_images.sort(key=lambda x: x.stat().st_size, reverse=True)

    # Calculate split indices
    num_images = len(all_images)
    num_train = int(num_images * 0.7)
    num_val = int(num_images * 0.15)

    # Step 3: Split the images into training, validation, and testing sets
    training_images = all_images[:num_train]
    validation_images = all_images[num_train:num_train + num_val]
    testing_images = all_images[num_train + num_val:]

    print(f"Copying {len(training_images)} images to training_data/{category}")
    print(f"Copying {len(validation_images)} images to validation_data/{category}")
    print(f"Copying {len(testing_images)} images to testing_data/{category}")

    # Step 4: Copy images to the respective output folders
    for img in training_images:
        shutil.copy(img, output_folders["train"][category])
    for img in validation_images:
        shutil.copy(img, output_folders["val"][category])
    for img in testing_images:
        shutil.copy(img, output_folders["test"][category])

# Process each category
for category in categories:
    input_folder = os.path.join(input_parent, category)
    organize_images(category, input_folder, output_dirs)

print("Images organized successfully!")
