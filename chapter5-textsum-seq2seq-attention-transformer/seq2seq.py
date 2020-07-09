import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import GRU, Bidirectional, Dense
import tensorflow_datasets as tfds

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import os
import time

class Embedding(object):
    embedding = None  # singleton

    @classmethod
    def get_embedding(self, vocab_size, embedding_dim):
        if self.embedding is None:
            self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                  mask_zero=True)
        return self.embedding

# Encoder
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        # embedding layer
        self.embedding = Embedding.get_embedding(vocab_size, embedding_dim)
        self.bigru = Bidirectional(GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform'),
                                       merge_mode='concat'
                                  )
        self.relu = Dense(self.enc_units, activation='relu')

    def call(self, x, hidden):
        x = self.embedding(x)  # We are using a mask

        output, forward_state, backward_state = self.bigru(x, initial_state = hidden)
        # now, concat the hidden states through the dense ReLU layer
        hidden_states = tf.concat([forward_state, backward_state], axis=1)
        output_state = self.relu(hidden_states)
        
        return output, output_state
        
    def initialize_hidden_state(self):
        return [tf.zeros((self.batch_size, self.enc_units)) for i in range(2)]



class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, decoder_hidden, enc_output):
        # decoder hidden state shape == (64, 256) [batch size, decoder units]
        # encoder output shape == (64, 128, 256) 
        # which is [batch size, max sequence length, encoder units]
        query = decoder_hidden # maps summarization to generic form of attention
        values = enc_output
        
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to broadcast addition along the time axis
        query_with_time_axis = tf.expand_dims(query, 1)
    
        # score shape == (batch_size, max_length, 1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))
    
        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
    
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
    
        return context_vector, attention_weights



class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        # Unique embedding layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                                  mask_zero=True)
        # Shared embedding layer
        #self.embedding = Embedding.get_embedding(vocab_size, embedding_dim)
        
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(vocab_size, activation='softmax', name='fc1')

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        x = self.embedding(x)
    
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    
        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))
        
        x = self.fc1(output)
        
        return x, state, attention_weights



loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
                    from_logits=False, reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file("gigaword32k.enc")
start = tokenizer.vocab_size + 1
end = tokenizer.vocab_size


def encode(article, summary, start=start, end=end, tokenizer=tokenizer, 
           art_max_len=128, smry_max_len=50):
    tokens = tokenizer.encode(article.numpy())
    if len(tokens) > art_max_len:
        tokens = tokens[:art_max_len]
    art_enc = sequence.pad_sequences([tokens], padding='post',
                                 maxlen=art_max_len).squeeze()
    
    tokens = [start] + tokenizer.encode(summary.numpy())
    
    if len(tokens) > smry_max_len:
        tokens = tokens[:smry_max_len]
    else:
        tokens = tokens + [end]
    
    smry_enc = sequence.pad_sequences([tokens], padding='post',
                                 maxlen=smry_max_len).squeeze()

    return art_enc, smry_enc


def tf_encode(article, summary):
    art_enc, smry_enc = tf.py_function(encode, [article, summary],
                                     [tf.int64, tf.int64])
    art_enc.set_shape([None])
    smry_enc.set_shape([None])
    return art_enc, smry_enc
