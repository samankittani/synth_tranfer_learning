import os
import shutil
from scripts.basemodel import train_base_model as tbs
from scripts.gensynthdata import gen_synth_data as gsd
from scripts.inithumandata import prep_human_data as phd


def empty_directory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.unlink(item_path)  # Removes files and links
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)  # Removes directories


# Directories to empty
dir1 = './datasets/synthdata/'
dir2 = './datasets/humandata/'

# Empty directories before running scripts
empty_directory(dir1)
empty_directory(dir2)

tbs()
gsd()
phd()
