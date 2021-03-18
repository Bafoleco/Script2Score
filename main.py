from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras import utils
from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import sigmoid
from tensorflow import keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import csv
import math
import numpy as np
import tensorflow as tf
import argparse
import os
from parse_scripts import get_direction_strs

# Custom activations for scores and revenues
def score_out(X):
   return 10 * sigmoid(X)

def revenue_out(X):
   return 23 * sigmoid(X)

# Based on https://keras.io/examples/nlp/pretrained_word_embeddings/
def create_word_embedding(word_index):

    # Load word embeddings 
    path_to_glove_file = "glove.6B.100d.txt"
    embeddings_index = {}
    with open(path_to_glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    # Prepare embedding matrix for our corpus
    num_tokens = len(word_index) + 2
    print(len(word_index))
    embedding_dim = 100
    hits = 0
    misses = 0
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for i, word in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))

    return embedding_matrix, num_tokens, embedding_dim


def get_data(using_revenues):

    # Load frequency data
    freq_vecs = {}
    bad_ids = []
    frequency_file = "frequency_csv.csv"
    with open(frequency_file, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            if len(row) < 1:
                continue
            id = row[0][: row[0].find('.')]
            if len(row) < 1000 or row[1] == 'UnicodeDecodeError':
                bad_ids.append(id)
                continue
            freq_vecs[id] = [int(idx) for idx in row[1: 251]]

    # Create a map from word ids to words 
    word_idx_file = "word_list.txt"
    with open(word_idx_file, 'r') as f:
        text = f.read()
        all_words = text.split(',')

    word_index = {}
    reverse_map = {}
    for i in range(len(all_words)):
        word_index[i] = all_words[i]
        reverse_map[all_words[i]] = i

    # Load script direction data
    direction_dict = get_direction_strs(reverse_map, len(word_index))

    # Load movie metadata
    metadata_file = "movie_metadata_updated.csv"
    data = []
    freq_data = []
    dir_data = []
    with open(metadata_file, 'r', encoding='iso-8859-1') as f:
        csv_reader = csv.reader(f)
        count = 0
        for row in csv_reader:
            try:
                id = "tt" + row[1].rjust(7, "0")

                if id not in bad_ids:
                    if "title" in row[0] or row[0] == "":
                        continue
                    train_example = row[2:]
                    try:
                        freq_data.append(freq_vecs[id])
                        data.append(train_example)
                        dir_data.append(direction_dict[id])
                    except KeyError:
                        print("key error")
                        continue
            except IndexError:
                pass

    #shuffle freq_data and data arrays in the same way
    np.random.seed(1)
    index_map = np.arange(len(data))
    np.random.shuffle(index_map)

    print(len(freq_data[0]))
    print(len(dir_data[0]))
    print(len(freq_data))
    print(len(dir_data))
    print(type(freq_data))
    print(type(dir_data))

    rand_data = [data[i] for i in index_map]
    rand_freqs = np.array([freq_data[i] for i in index_map]).astype(np.float).T
    rand_dir_data = np.array([dir_data[i] for i in index_map]).astype(np.float).T

    print(rand_dir_data.shape)
    print(rand_freqs.shape)

    # Select the input array
    X = np.array(rand_data)[:,:-2].T.astype(np.float)

    # Select optimization target based on command line args
    Y = np.array(rand_data)[np.newaxis,:,-1].astype(np.float)  
    if using_revenues:
        Y = np.array(rand_data)[np.newaxis,:,-2].astype(np.float)

    # Seperation of dataset into train, dev, and test sets
    X_train = X[:,:(math.floor(.8 * X.shape[1]))]
    freq_train = rand_freqs[:,:(math.floor(.8 * X.shape[1]))]
    dir_train = rand_dir_data[:,:(math.floor(.8 * X.shape[1]))]
    y_train = Y[:,:(math.floor(.8 * X.shape[1]))]

    X_dev = X[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    freq_dev = rand_freqs[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    dir_dev = rand_dir_data[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    y_dev = Y[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]

    X_test = X[:,(math.floor(.9 * X.shape[1])):]
    freq_test = rand_freqs[:,(math.floor(.9 * X.shape[1])):]
    dir_test = rand_dir_data[:,(math.floor(.9 * X.shape[1])):]
    y_test = Y[:,(math.floor(.9 * X.shape[1])):]

    return word_index, X_train, y_train, freq_train, dir_train, X_dev, freq_dev, dir_dev, y_dev, X_test, freq_test, dir_test, y_test

def load_data(X, freq, direction):
    return {"numeric": X.T[:, 114:117], "categorical":  X.T[:,:114], "most_common_words": freq.T, "direction_subset": direction.T}
    # return {"numeric": X.T[:, 114:117], "categorical":  X.T[:,:114], "most_common_words": freq.T}

def setup_numeric(inputs, features):
    numeric_input = keras.Input(shape=(3,), name="numeric")
    # Add inputs and features 
    inputs.append(numeric_input)
    features.append(numeric_input)

def setup_categorical(inputs, features):
    categorical_input = keras.Input(shape=(114,), name="categorical")
    #create categorical features
    categorical_features = Dense(128, activation='linear', use_bias=False)(categorical_input)
    # Add inputs and features 
    inputs.append(categorical_input)
    features.append(categorical_features)

def setup_frequency(inputs, features, embedding_layer):
    frequency_input = keras.Input(shape=(250,), name="most_common_words")
    
    freq_features = embedding_layer(frequency_input)
    freq_features = Flatten()(freq_features)

    #preprocess word embeddings
    freq_features = BatchNormalization()(freq_features)
    freq_features = Dense(1024, kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(freq_features)
    freq_features = Dense(256, kernel_regularizer=regularizers.l2(l2_param),  activation='tanh')(freq_features)

    inputs.append(frequency_input)
    features.append(freq_features)

def setup_direction(inputs, features, embedding_layer):
    direction_input = keras.Input(shape=(150,), name="direction_subset")

    dir_features = embedding_layer(direction_input)
    
    print(dir_features)

    #preprocess word embeddings
    dir_features = layers.LSTM(1, return_sequences=False)(dir_features)
    print(dir_features)

    inputs.append(direction_input)
    features.append(dir_features)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--rev', action='store_true')
    parser.add_argument('--num', action='store_true')
    parser.add_argument('--cat', action='store_true')
    parser.add_argument('--freq', action='store_true')
    parser.add_argument('--dir', action='store_true')

    args = parser.parse_args()

    word_index, X_train, y_train, freq_train, dir_train, X_dev, freq_dev, dir_dev, y_dev, X_test, freq_test, dir_test, y_test = get_data(args.rev)

    #Set output activation based on whether we are predicting log revenues or scores
    if args.rev:
        utils.get_custom_objects().update({'custom_activation': Activation(revenue_out)})
    else: 
        utils.get_custom_objects().update({'custom_activation': Activation(score_out)})

    l2_param = 0.005

    # Prepare inputs and features 
    inputs = []
    features = []
    
    if args.num:
        setup_numeric(inputs, features)
    if args.cat:
        setup_categorical(inputs, features)
    
    #script based features
    if args.freq or args.dir:
        embedding_matrix, num_tokens, embedding_dim = create_word_embedding(word_index)
        embedding_layer = Embedding(
            num_tokens,
            embedding_dim,
            embeddings_initializer=keras.initializers.Constant(embedding_matrix),
            trainable=False,
        )
    if args.freq:
        setup_frequency(inputs, features, embedding_layer)
    if args.dir:
        setup_direction(inputs, features, embedding_layer)

    #concatenate all features
    if len(features) > 1:
        x = layers.concatenate(features)
    else: 
        x = features[0]

    #fully connected network 
    x = BatchNormalization()(x)
    x = Dense(256,  kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(x)
    x = Dense(64,  kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(x)
    outputs = Dense(1,  kernel_regularizer=regularizers.l2(l2_param), activation='custom_activation')(x)

    # Create model
    optimizer = keras.optimizers.Adam(lr=0.0002)
    model = keras.Model(inputs, outputs, name="Script2Score")

    # We use early stopping to help prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_mean_squared_error",
        min_delta=0,
        patience=3000,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    # Compile model
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    # Train model
    model.fit(
              load_data(X_train, freq_train, dir_train),
              y_train.T,
              batch_size=500,
              epochs=40,
            #   callbacks=[early_stopping],
              verbose=1,
              validation_data=(load_data(X_dev, freq_dev, dir_dev), y_dev.T)
         )

    print(model.predict(load_data(X_dev, freq_dev, dir_dev)))

    # Summary of neural network, saved in 'model' folder
    model.summary()

    score = model.evaluate(load_data(X_dev, freq_dev, dir_dev), y_dev.T)
    print("Dev set results:")
    print(score)

    score = model.evaluate(load_data(X_test, freq_test, dir_test), y_test.T)
    print("Test set results:")
    print(score)

    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    model.save('model')
