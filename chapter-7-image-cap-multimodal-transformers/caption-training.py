import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import tensorflow_datasets as tfds

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import json
from glob import glob
from PIL import Image
import pickle
import re
import os
import time
import datetime

from tqdm import tqdm
# our visual transformer code
import visual_transformer as vt

####### GPU CONFIGS FOR RTX 2070/NVidia GPU ###############
## Please comment out if not training on GPU  ##
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

#########################
# Load Data file mapping captions to images
#########################
prefix = './data/'
save_prefix = prefix + "features/"  # for storing prefixes
annot = prefix + 'data.csv'

inputs = pd.read_csv(annot, header=None, names=["caption", "image"])
print("Data file loaded")

#########################
# Tokenize Captions
#########################
cap_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(
    "captions")
print(cap_tokenizer.encode("A man riding a wave on top of a surfboard.".lower()))
print("Tokenizer hydrated")

# Max length of captions split by spaces
lens = inputs['caption'].map(lambda x: len(x.split()))

# Max length of captions after tokenization
# tfds demonstrated in earlier chapters
# This is a quick way if data fits in memory
lens = inputs['caption'].map(lambda x: len(cap_tokenizer.encode(x.lower())))

# We will set this as the max length of captions
# which cover 99% of the captions without truncation
max_len = int(lens.quantile(0.99) + 1)  # for special tokens

start = '<s>'
end = '</s>'
inputs['tokenized'] = inputs['caption'].map(
    lambda x: start + x.lower().strip() + end)
print("Some prepared captions: ", inputs.tokenized[:5])


def tokenize_pad(x):
    x = cap_tokenizer.encode(x)
    if len(x) < max_len:
        x = x + [0] * int(max_len - len(x))
    return x[:max_len]


inputs['tokens'] = inputs.tokenized.map(lambda x: tokenize_pad(x))
print("Captions tokenized and padded/truncated")

# now to compute a column with the new name of the saved image feature file
inputs['img_features'] = inputs['image'].map(lambda x:
                                             save_prefix +
                                             x.split('/')[-1][:-3]
                                             + 'npy')

#########################
# Prepare tf.DataSet for training
#########################

captions = inputs.tokens.tolist()
img_names = inputs.img_features.tolist()

# we only took half validation examples so we dont need to split
# img_train, img_val, cap_train, cap_val = train_test_split(img_names,
#                                                          captions,
#                                                          test_size=0.2,
#                                                          random_state=42)
img_train, cap_train = img_names, captions

# Load the numpy file with extracted ResNet50 feature

def load_image_feature(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8'))
    return img_tensor, cap


dataset = tf.data.Dataset.from_tensor_slices((img_train, cap_train))

# Use map to load the numpy files in parallel
dataset = dataset.map(lambda item1, item2: tf.numpy_function(
    load_image_feature, [item1, item2], [tf.float32, tf.int32]),
    num_parallel_calls=tf.data.experimental.AUTOTUNE)

# To verify
for img, cap in dataset.take(2):
    print(img.shape)
    print(cap.numpy())
print("Training dataset prepared.")

#########################
# Build Transformer Model
#########################
# These parameters control the size and complexity of the model
# BERT (base) uses 12 layers, 768 as embedding dim, 12 attention heads
# and 4H (4x768) as feedforward size

# Small Model
num_layers = 4
d_model = 128
dff = d_model * 4
num_heads = 8


# BERT Base Model
# num_layers = 12
# d_model = 768
# dff = d_model * 4  # as per BERT paper
# num_heads = 12

target_vocab_size = cap_tokenizer.vocab_size  # already includes start/end tokens
dropout_rate = 0.1

EPOCHS = 20  # should see results in 4-10 epochs also

transformer = vt.Transformer(num_layers, d_model, num_heads, dff,
                             target_vocab_size,
                             pe_input=49,  # 7x7 pixels
                             pe_target=target_vocab_size,
                             rate=dropout_rate,
                             use_pe=False
                             )

#########################
# Training Setup
#########################
# Learning Rate Schedule, as per `Attention is All You Need' paper
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                    epsilon=1e-9)


# Visualize the schedule: uncomment to plot
# import matplotlib.pyplot as plt
# temp_learning_rate_schedule = CustomSchedule(d_model)
#
# plt.plot(temp_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
# plt.ylabel("Learning Rate")
# plt.xlabel("Train Step")

#########################
# Loss and Metrics
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')

#########################
# Helper function for creating masks

def create_masks(inp, tar):
    # Encoder padding mask - This should just be 1's
    # input shape should be (batch_size, 49, 2048)
    inp_seq = tf.ones([inp.shape[0], inp.shape[1]])  # all pixels to be used

    enc_padding_mask = vt.create_padding_mask(inp_seq)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = vt.create_padding_mask(inp_seq)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = vt.create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = vt.create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


# Checkpoints setup
checkpoint_path = "./checkpoints/train-small-model-nope-20ep"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, 
                                          max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

#########################
# Training Loops
#########################
# setup training parameters
BUFFER_SIZE = 1000
BATCH_SIZE = 64  # can reduce or increase depending on GPU capacity
# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Perform one step of raining on one batch in an epoch

@tf.function
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp,
                                                                     tar_inp)

    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    train_accuracy(tar_real, predictions)


# Begin Training
for epoch in range(EPOCHS):
    start_tm = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    # inp -> images, tar -> caption
    for (batch, (inp, tar)) in enumerate(dataset):
        train_step(inp, tar)

        if batch % 100 == 0:
            ts = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
            print('[{}] Epoch {} Batch {} Loss {:.6f} Accuracy {:.6f}'.format(
                ts, epoch + 1, batch, train_loss.result(),
                train_accuracy.result()))

    if (epoch + 1) % 2 == 0:
        ckpt_save_path = ckpt_manager.save()
        print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                            ckpt_save_path))

    print('Epoch {} Loss {:.6f} Accuracy {:.6f}'.format(epoch + 1,
                                                        train_loss.result(),
                                                        train_accuracy.result()))

    print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start_tm))
    
transformer.summary()
