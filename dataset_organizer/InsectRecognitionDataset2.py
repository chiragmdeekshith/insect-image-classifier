import os
import shutil
from pathlib import Path
import tempfile

# Define input and output directories
input_parent = "../Datasets/InsectRecognition/"
categories = ["Butterfly", "Dragonfly", "Grasshopper", "Ladybird", "Mosquito"]
output_parent = "../organized_datasets/InsectRecognition/"

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
def organize_images(category, input_folder, output_folders, temp_dir):
    all_images = []

    # Step 1: Copy all images to temp_dir and gather their paths
    for img_path in Path(input_folder).glob("*"):
        if img_path.is_file():
            copied_img = shutil.copy(img_path, temp_dir)
            all_images.append(Path(copied_img))

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

    # Step 4: Move images to the respective output folders
    for img in training_images:
        shutil.copy(img, output_folders["train"][category])
    for img in validation_images:
        shutil.copy(img, output_folders["val"][category])
    for img in testing_images:
        shutil.copy(img, output_folders["test"][category])


# Create a temporary directory for processing
with tempfile.TemporaryDirectory() as temp_dir:
    # Process each category
    for category in categories:
        input_folder = os.path.join(input_parent, category)
        organize_images(category, input_folder, output_dirs, temp_dir)

print("Images organized successfully!")