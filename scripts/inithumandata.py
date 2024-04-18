import os
import shutil
import random


def move_perc_to_human_data(source_dir, destination_dir, percent):

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # List all files in the source directory
    files = [file for file in os.listdir(
        source_dir) if os.path.isfile(os.path.join(source_dir, file))]

    move_len = len(files) * percent
    # Check if there are at least 225 images
    # Randomly select 225 images
    selected_files = random.sample(files, int(move_len))

    # Move the selected images to the destination directory
    for file in selected_files:
        original_path = os.path.join(source_dir, file)

        # Add '_1' before the file extension
        new_filename = file.rsplit(
            '.', 1)[0] + '_1.' + file.rsplit('.', 1)[1]
        new_path = os.path.join(destination_dir, new_filename)
        shutil.move(original_path, new_path)
    print(f"Moved {len(selected_files)} images to {destination_dir}")


def prep_human_data():
    move_perc_to_human_data('datasets/synthdata/RIGHT/',
                            'datasets/humandata/', .30)

    move_perc_to_human_data('datasets/synthdata/WRONG/',
                            'datasets/humandata/', .30)
