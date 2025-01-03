import os
import shutil
from pathlib import Path
import tempfile

# Define paths for input folders and output structure
input_parent = "../Datasets/BeeOrWasp/"
bee1 = input_parent + "bee1"
bee2 = input_parent + "bee2"
wasp1 = input_parent + "wasp1"
wasp2 = input_parent + "wasp2"
input_dirs = [bee1, bee2, wasp1, wasp2]

output_parent = "../organized_datasets/BeeOrWasp/"
output_dirs = {
    "bee": {
        "train": output_parent + "training_data/bee",
        "val": output_parent + "validation_data/bee",
        "test": output_parent + "testing_data/bee"
    },
    "wasp": {
        "train": output_parent + "training_data/wasp",
        "val": output_parent + "validation_data/wasp",
        "test": output_parent + "testing_data/wasp"
    }
}

# Create output directories if they don't exist
for species in output_dirs:
    for folder in output_dirs[species].values():
        os.makedirs(folder, exist_ok=True)

# Function to merge, organize, and copy images
def organize_images(species, input_folders, output_folders, temp_dir):
    # Step 1: Gather all images in input folders and copy them to the temp_dir
    all_images = []
    for folder in input_folders:
        for img_path in Path(folder).glob("*"):
            if img_path.is_file():
                # Copy image to temporary directory and add to list
                copied_img = shutil.copy(img_path, temp_dir)
                all_images.append(Path(copied_img))

    # Step 2: Sort images by descending order of file size
    all_images.sort(key=lambda x: x.stat().st_size, reverse=True)

    # Step 3: Select the top 1500 images for training
    training_images = all_images[:1500]

    # Step 4: Move bottom 225 from training_images to testing folder
    testing_images = training_images[-225:]
    for img in testing_images:
        shutil.move(str(img), output_folders["test"])

    # Step 5: Move the next bottom 225 from training  to validation folder
    validation_images = training_images[-450:-225]
    for img in validation_images:
        shutil.move(str(img), output_folders["val"])

    # Remaining images go to the training folder
    remaining_training_images = training_images[:-450]
    for img in remaining_training_images:
        shutil.move(str(img), output_folders["train"])

# Create a temporary directory for copying images
with tempfile.TemporaryDirectory() as temp_dir:
    # Run the process for both bee and wasp
    organize_images("bee", [bee1, bee2], output_dirs["bee"], temp_dir)
    organize_images("wasp", [wasp1, wasp2], output_dirs["wasp"], temp_dir)

print("Images organized successfully!")
