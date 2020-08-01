# Import
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import json
from PIL import Image
import pickle
import re
import os
import time
import datetime
import csv
import random

# Please do steps 1-4 in terminal
# Step 1: Download the Datasets
# Download Training data from COCO captions 2014
# This is a 13GB download!!!
# wget http://images.cocodataset.org/zips/train2014.zip

# this is a 6GB download - validation set
# wget http://images.cocodataset.org/zips/val2014.zip

# get the annotations/captions - 214MB
# wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip

# Step 2: Unzip and move files
# unzip train2014.zip
# unzip val2014.zip
# unzip annotations_trainval2014.zip

# Step 3: Move all data into pretrain folder for processing
# mkdir pretrain
# mv train2014 pretrain/
# mv val2014 pretrain/
# mv annotations pretrain/

# Step 4: Process JSON annotation files
valcaptions = json.load(open(
    './data/annotations/captions_val2014.json', 'r'))

trcaptions = json.load(open(
    './data/annotations/captions_train2014.json', 'r'))

# inspect the annotations
print(trcaptions.keys())

prefix = "./data/"
val_prefix = prefix + 'val2014/'
train_prefix = prefix + 'train2014/'

# training images
trimages = {x['id']: x['file_name'] for x in trcaptions['images']}
# validation images
# take half images from validation - karpathy split
valset = len(valcaptions['images']) - 5000 # leave last 5k 
valimages = {x['id']: x['file_name'] for x in valcaptions['images'][:valset]}
truevalimg = {x['id']: x['file_name'] for x in valcaptions['images'][valset:]}

# we flatten to (caption, image_path) structure
data = list()
errors = list()
validation = list()

for item in trcaptions['annotations']:
    if int(item['image_id']) in trimages:
        fpath = train_prefix + trimages[int(item['image_id'])]
        caption = item['caption']
        data.append((caption, fpath))
    else:
        errors.append(item)

for item in valcaptions['annotations']:
    caption = item['caption']
    if int(item['image_id']) in valimages:
        fpath = val_prefix + valimages[int(item['image_id'])]
        data.append((caption, fpath))
    elif int(item['image_id']) in truevalimg: # reserved
        fpath = val_prefix + truevalimg[int(item['image_id'])]
        validation.append((caption, fpath))
    else:
        errors.append(item)                  

random.seed(42)  # for reproducibility

# lets shuffle the list in place
print("Before Shuffling: ", data[:5])
random.shuffle(data)
print("Post-shuffling: ", data[:5])

# persist for future use
with open(prefix + 'data.csv', 'w') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerows(data)

# persist for future use
with open(prefix + 'validation.csv', 'w') as file:
    writer = csv.writer(file, quoting=csv.QUOTE_ALL)
    writer.writerows(validation)

print("TRAINING: Total Number of Captions: {},  Total Number of Images: {}".format(
    len(data), len(trimages) + len(valimages)))

print("VALIDATION/TESTING: Total Number of Captions: {},  Total Number of Images: {}".format(
    len(validation), len(truevalimg)))

print("Errors: ", errors)