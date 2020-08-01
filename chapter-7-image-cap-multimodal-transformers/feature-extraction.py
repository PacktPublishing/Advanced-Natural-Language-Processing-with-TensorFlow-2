'''
Extract features from ResNet50 and store them on
disk for retreival during training. This makes training faster
and allows more efficient use of GPU resources. However, it
impacts accuracy as the ResNet50 pre-trained model cant be
fine-tuned for this task.
'''
# Imports
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import tensorflow_datasets as tfds

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

from tqdm import tqdm

####### GPU CONFIGS FOR RTX 2070 ###############
## Please ignore if not training on GPU       ##
## this is important for running CuDNN on GPU ##

tf.keras.backend.clear_session()  # - for easy reset of notebook state

# chck if GPU can be seen by TF
tf.config.list_physical_devices('GPU')
# tf.debugging.set_log_device_placement(True)  # only to check GPU usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
###############################################

prefix = './data/'
save_prefix = prefix + "features/"  # for storing prefixes
annot = prefix + 'data.csv'
# load the pre-processed file
inputs = pd.read_csv(annot, header=None, names=["caption", "image"])

# We are going to use the last residual block of the ResNet50 architecture
# which has dimension 7x7x2048 and store into individual file

def load_image(image_path, size=(224, 224)):
    # pre-processes images for ResNet50 in batches 
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size)
    image = preprocess_input(image)  # from keras.applications.ResNet50
    return image, image_path


uniq_images = sorted(inputs['image'].unique())  # only process unique images
print("Unique images: ", len(uniq_images))

image_dataset = tf.data.Dataset.from_tensor_slices(uniq_images)
image_dataset = image_dataset.map(
    load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

rs50 = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",  # no pooling
    input_shape=(224, 224, 3)
)

new_input = rs50.input
hidden_layer = rs50.layers[-1].output

features_extract = tf.keras.Model(new_input, hidden_layer)
features_extract.summary()

save_prefix = prefix + "features/"
try:
    # Create this directory : mkdir {save_prefix}
    os.mkdir(save_prefix)
except FileExistsError:
    pass # Directory already exists

for img, path in tqdm(image_dataset):
    batch_features = features_extract(img)
    batch_features = tf.reshape(batch_features,
                                (batch_features.shape[0], -1, batch_features.shape[3]))

    for feat, p in zip(batch_features, path):
        filepath = p.numpy().decode("utf-8")
        filepath = save_prefix + filepath.split('/')[-1][:-3] + "npy"
        np.save(filepath, feat.numpy())

print("Images saved as npy files")

# Now, read the labels and create a subword tokenizer with it
# ~8K vocab size
cap_tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    inputs['caption'].map(lambda x: x.lower().strip()).tolist(),
    target_vocab_size=2**13, reserved_tokens=['<s>', '</s>'])
cap_tokenizer.save_to_file("captions")
