import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

import tensorflow_datasets as tfds 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import json
from glob import glob
from PIL import Image
import pickle
import re
import os
import time, datetime

from tqdm import tqdm

####### GPU CONFIGS FOR RTX 2070 ###############
## Please ignore if not training on GPU       ##
## this is important for running CuDNN on GPU ##

tf.keras.backend.clear_session() #- for easy reset of notebook state

# chck if GPU can be seen by TF
tf.config.list_physical_devices('GPU')
#tf.debugging.set_log_device_placement(True)  # only to check GPU usage
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

inputs = pd.read_csv(annot, header=None, names=["caption", "image"])

cap_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file("captions")

start = '<s>'
end = '</s>'
inputs['tokenized'] = inputs['caption'].map(lambda x: start+x.lower().strip()+end)
inputs.tokenized[:5]

max_len = 22  # added one for start token
def tokenize_pad(x):
    x = cap_tokenizer.encode(x)
    if len(x) < max_len:
        x = x + [0] * (max_len - len(x))
    return x[:max_len]

inputs['tokens'] = inputs.tokenized.map(lambda x: tokenize_pad(x))

# now to compute a column with the new name of the saved image feature file
#inputs['img_features'] = inputs['image'].map(lambda x: 
#                                             save_prefix + 
#                                             x.split('/')[-1][:-3] 
#                                             + 'npy')
#
# create a train/test split
captions = inputs.tokens.tolist()
img_names = inputs.image.tolist()  # raw image names

# we only took half validation examples so we dont need to split
#img_train, img_val, cap_train, cap_val = train_test_split(img_names,
#                                                          captions,
#                                                          test_size=0.2,
#                                                          random_state=42)
#
img_train, cap_train = img_names, captions
# Note that there is leakage as each image has multiple captions
print(len(img_train), len(cap_train))

# pass images through 
def load_image_captions(image_path, captions, size=(224, 224)):  
    #print(image_path, captions)
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, size)
    image = preprocess_input(image) #from ResNet50
    return image, captions #image_path

dataset = tf.data.Dataset.from_tensor_slices((img_train, cap_train))

# fix as per issue: https://github.com/tensorflow/tensorflow/issues/29931
temp = tf.zeros([4, 224, 224, 3])
tf.keras.applications.resnet50.preprocess_input(temp)


dataset = dataset.map(load_image_captions, 
                       num_parallel_calls=tf.data.experimental.AUTOTUNE) 

# To verify
for img, cap in dataset.take(2):
    print(img.shape)
    print(cap.numpy())
    
 
#########################
# Positional Encoder
#########################

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

#########################
# Mask Calculations
#########################

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# while decoding, we dont have recurrence and dont want decoder
# to see tokens from the future

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

#########################
# Attention
#########################

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(
        scaled_attention_logits,
        axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)

        return output, attention_weights

#########################
# Feed Forward Layer
#########################
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


#########################
# Visual Encoder
#########################
# Encoder layer is identical to standard transformer
# However, Encoder processes and embeds images differently than
# text. Code is VisualEncoder has been modified for image captioning

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # (batch_size, input_seq_len, d_model)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection

        return out2

# This is unique to our Image Captioning problem
class VisualEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 maximum_position_encoding=49, dropout_rate=0.1):
        # we have 7x7 images from ResNet50, and each pixel is an input token
        # which has been embedded into 2048 dimensions by ResNet
        super(VisualEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        
        # FC layer replaces embedding layer in traditional encoder
        # this FC layers takes 49x2048 image and projects into model dims
        self.fc = tf.keras.layers.Dense(d_model, activation='relu')
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        # all inp image sequences are always 49, so mask not needed
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        # input size should be batch_size, 49, 2048)
        x = self.fc(x)  # output dims should be (batch_size, 49, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  # scaled dot product attention
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](
                x, training, mask)  # mask shouldnt be needed

        return x  # (batch_size, 49, d_model)

# This is unique to our Image Captioning problem
class VisualResNetEncoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                   maximum_position_encoding=49, dropout_rate=0.1):
        # we have 7x7 images from ResNet50, and each pixel is an input token
        # which has been embedded into 2048 dimensions by ResNet
        super(VisualResNetEncoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        # add a resNet50 as a layer
        rs50 = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights="imagenet",  #no pooling
                    input_shape=(224, 224, 3)
                )
        
        self.features_extract = rs50
        
        self.fc = tf.keras.layers.Dense(d_model, activation='relu')
        self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                                self.d_model)
        
        
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) 
                           for _ in range(num_layers)]
    
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def train_resnet(self, yes=True):
        # make all the laers of the resnet trainable or not
        for layer in self.features_extract.layers:
                layer.trainable = yes # boolean
                
        
    def call(self, x, training, mask):
        # we need to convert 224x224x3 images in 49x2048
        x = self.features_extract(x)
        x = tf.reshape(x,
                    (x.shape[0], -1, x.shape[3])) # 49x2048 now
        
        #all inp image sequences are always 49, so mask not needed
        seq_len = tf.shape(x)[1] 
        
        #x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        
        # adding embedding and position encoding.
        # input size should be batch_size, 49, 2048)
        x = self.fc(x)  # output dims should be (batch_size, 49, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)
        
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)  # mask shouldnt be needed
        
        return x  # (batch_size, 49, d_model)

#########################
# Decoder
#########################

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)

        attn1, attn_weights_block1 = self.mha1(
            x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x) # residual

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2


class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(
                                maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                                   look_ahead_mask, 
                                                   padding_mask)

        attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

#########################
# Full Transformer
#########################

class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, 
             target_vocab_size, pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = VisualResNetEncoder(num_layers, d_model, num_heads, dff, 
                                     pe_input, rate)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                             target_vocab_size, pe_target, rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inp, tar, training, enc_padding_mask, 
         look_ahead_mask, dec_padding_mask):

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
          tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights

# These parameters control the size and complexity of the model
# BERT (base) uses 12 layers, 768 as embedding dim, 12 attention heads
# and 4H (4x768) as feedforward size
# Small Model
num_layers = 4
d_model = 128
dff = 512
num_heads = 8

target_vocab_size = cap_tokenizer.vocab_size # already includes start/end tokens
dropout_rate = 0.1

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          target_vocab_size, 
                          pe_input=49, # 7x7 pixels
                          pe_target=target_vocab_size,
                          rate=dropout_rate
                          )
# we dont want ResNet50 to train from the start
transformer.encoder.train_resnet(False)
print("Starting training with ResNet not trainable")

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

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
    name='train_accuracy')


def create_masks(inp, tar):
    # Encoder padding mask - This should just be 1's
    # input shape should be (batch_size, 49, 2048)
    # inp_seq = tf.ones([inp.shape[0], inp.shape[1]]) # all pixels to be used
    inp_seq = tf.ones([inp.shape[0], 49])  # as resnet is now part of architecture
    # cant use input shapes any more as they should be 224x224x3
    enc_padding_mask = create_padding_mask(inp_seq)
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp_seq)
    
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask


checkpoint_path = "./checkpoints/train-small-model-40ep"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print ('Latest checkpoint restored!!')

    
@tf.function 
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
    # tar_inp.shape  ==(batch_size, 22-1)
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

EPOCHS=20
BUFFER_SIZE=1000
BATCH_SIZE=64
# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


for epoch in range(EPOCHS):
    start = time.time()
    
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    # we are going to train 8 epochs without resnet and last 14 epochs
    # we will start training resnet layers
    if epoch == 17:
        transformer.encoder.train_resnet(True)
        print("Now the ResNet layer will be fine tuned")
        
    # inp -> images, tar -> caption
    for (batch, (inp, tar)) in enumerate(dataset):
        train_step(inp, tar)
        
        if batch % 100 == 0:
            ts = datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S)")
            print ('[{}] Epoch {} Batch {} Loss {:.6f} Accuracy {:.6f}'.format(
                ts, epoch + 1, batch, train_loss.result(), 
                train_accuracy.result()))
          
    if (epoch + 1) % 2 == 0:
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                             ckpt_save_path))
        
    print ('Epoch {} Loss {:.6f} Accuracy {:.6f}'.format(epoch + 1, 
                                                  train_loss.result(), 
                                                  train_accuracy.result()))

    print ('Time taken for 1 epoch: {:.2f} secs\n'.format(time.time() - start))
    
transformer.summary()