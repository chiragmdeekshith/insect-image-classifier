import os
import re

def get_highest_training_accuracy(base_dir, dataset_range=(1, 6)):
    highest_accuracy = 0
    best_folder = ""

    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(base_dir):
        # Only include folders that end with dataset_X within the specified range
        if not any(root.endswith(f"dataset_{i}") for i in range(dataset_range[0], dataset_range[1] + 1)):
            continue

        if "training.log" in files:
            log_path = os.path.join(root, "training.log")

            # Open the training log and search for accuracy lines
            with open(log_path, "r") as log_file:
                for line in log_file:
                    # Use regex to find accuracy values (example pattern: "Test Accuracy: 0.95")
                    match = re.search(r"Test Accuracy: (\d*\.\d+)", line)
                    if match:
                        accuracy = float(match.group(1))
                        if accuracy > highest_accuracy:
                            highest_accuracy = accuracy
                            best_folder = root

    return best_folder, highest_accuracy

# Specify the base directory to search
base_directory = "../mobilenet/pretrained_experiments_results/"  # Change this to your folder's path if necessary

# Set the dataset number range (inclusive) to search
dataset_range = (6,6)  # This will search in dataset_1 to dataset_6

best_folder, highest_accuracy = get_highest_training_accuracy(base_directory, dataset_range)
print(f"The folder with the highest testing accuracy is: {best_folder}")
print(f"Highest testing accuracy: {highest_accuracy}")
