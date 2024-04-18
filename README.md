# Code Review
There are several files in this repository. 

## Dependencies
numpy
matplotlib
tensorflow
keras
etc.

## Main.py
To test the transfer model, simply run main.py.

## baseline.py
This trains a model directly on the human labeled data. simply run the file.  

## Scripts
This directory includes several scripts that are used to generate the synthetic data and prep data for human labeling.
These scripts will delete the old human labeled data. 

## Setup.py
DO **NOT** run the setup.py script. The setup.py script deletes all the synthetic and human  labeled data.
It will then retrain the base model, 
regenerate synthetic data, and place 30% of the synthetic data into the datasets/humandata/ directory.
You will then have to use the datasets/utils/label_helper.sh script inside the human datadirectory to hand label the data.
**AVOID** this. Instead, I pushed the synthetic and human labeled data to try.
