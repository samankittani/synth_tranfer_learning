# Code Review
There are several files in this repository. 

## Dependencies
There are several dependencies. A complete list is available in the dependencies text file. THe main packages include tensorflow, matplotlib, and numpy

## Main.py
To test the transfer model, simply run `python3 main.py`. This will use the data in `datasets/synthdata` and `datasets/maindata` to train the model.

## baseline.py
This trains a model directly on the human labeled data. simply run the file. `python3 baseline.py`

## Scripts
This directory includes several scripts that are used to generate the synthetic data and prep data for human labeling.
These scripts will delete the old human labeled data. Feel free to explore them. 

## Setup.py
DO **NOT** run the setup.py script. The setup.py script deletes all the synthetic and human  labeled data.
It will then retrain the base model, 
regenerate synthetic data, and place 30% of the synthetic data into the datasets/humandata/ directory.
You will then have to use the datasets/utils/label_helper.sh script inside the human datadirectory to hand label the data.
**AVOID** this. Instead, I pushed sample synthetic and human labeled data to try.
