import os
import shutil
import random

source_dir = 'synthdata/RIGHT/'  # Source directory containing the images
# Destination directory for the selected images
destination_dir = 'humandata/'

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# List all files in the source directory
files = [file for file in os.listdir(
    source_dir) if os.path.isfile(os.path.join(source_dir, file))]

# Check if there are at least 225 images
if len(files) < 225:
    print("Error: Not enough images to select from.")
else:
    # Randomly select 225 images
    selected_files = random.sample(files, 225)

    # Move the selected images to the destination directory
    for file in selected_files:
        original_path = os.path.join(source_dir, file)
        # Add '_1' before the file extension
        new_filename = file.rsplit('.', 1)[0] + '_1.' + file.rsplit('.', 1)[1]
        new_path = os.path.join(destination_dir, new_filename)
        shutil.move(original_path, new_path)
    print(f"Moved {len(selected_files)} images to {destination_dir}")

source_dir = 'synthdata/WRONG/'  # Source directory containing the images

# List all files in the source directory
files = [file for file in os.listdir(
    source_dir) if os.path.isfile(os.path.join(source_dir, file))]

# Check if there are at least 30 images
if len(files) < 30:
    print("Error: Not enough images to select from.")
else:
    # Randomly select 30 images
    selected_files = random.sample(files, 30)

    # Move the selected images to the destination directory
    for file in selected_files:
        original_path = os.path.join(source_dir, file)
        # Add '_0' before the file extension
        new_filename = file.rsplit('.', 1)[0] + '_0.' + file.rsplit('.', 1)[1]
        new_path = os.path.join(destination_dir, new_filename)
        shutil.move(original_path, new_path)
    print(f"Moved {len(selected_files)} images to {destination_dir}")
