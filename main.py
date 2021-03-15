import csv
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras import utils
from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import sigmoid
from tensorflow import keras as keras
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import os


#custom activation to be used
def scaled_sigmoid(X):
   return 10 * sigmoid(X)

# based on https://keras.io/examples/nlp/pretrained_word_embeddings/
def create_word_embedding(word_index):

    path_to_glove_file = "glove.6B.100d.txt"

    embeddings_index = {}

    with open(path_to_glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    num_tokens = len(word_index) + 2
    print(len(word_index))
    embedding_dim = 100
    hits = 0
    misses = 0

    # Prepare embedding matrix
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


def get_data():
    frequency_file = "frequency_csv.csv"
    freq_vecs = {}
    bad_ids = []

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

    word_idx_file = "word_list.txt"

    with open(word_idx_file, 'r') as f:
        text = f.read()
        all_words = text.split(',')

    word_index = {}
    for i in range(len(all_words)):
        word_index[i] = all_words[i]

    metadata_file = "movie_metadata_updated.csv"
    data = []
    freq_data = []
    with open(metadata_file, 'r', encoding='iso-8859-1') as f:
        csv_reader = csv.reader(f)
        count = 0
        for row in csv_reader:
            try:
                id = "tt" + row[1].rjust(7, "0")

                if id not in bad_ids:
                    if "title" in row[0] or row[0] == "":
                        print(row[113])
                        continue
                    train_example = row[2:]
                    try:
                        freq_data.append(freq_vecs[id])
                        data.append(train_example)
                    except KeyError:
                        print("key error")
                        continue

            except IndexError:
                pass

    np.random.seed(1)

    index_map = np.arange(len(data))
    np.random.shuffle(index_map)

    rand_data = [data[i] for i in index_map]
    rand_freqs = np.array([freq_data[i] for i in index_map]).astype(np.float).T

    X = np.array(rand_data)[:,:-2].T.astype(np.float)
    Y = np.array(rand_data)[np.newaxis,:,-1].astype(np.float)  # Change to -2 for revenues instead of rating (-1)

    print(X.shape)
    print(Y.shape)

    #seperation of dataset into train and test sets
    X_train = X[:,:(math.floor(.8 * X.shape[1]))]
    freq_train = rand_freqs[:,:(math.floor(.8 * X.shape[1]))]

    y_train = Y[:,:(math.floor(.8 * X.shape[1]))]

    X_dev = X[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    freq_dev = rand_freqs[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    y_dev = Y[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]

    X_test = X[:,(math.floor(.9 * X.shape[1])):]
    freq_test = rand_freqs[:,(math.floor(.9 * X.shape[1])):]
    y_test = Y[:,(math.floor(.9 * X.shape[1])):]

    return word_index, X_train, y_train, freq_train, X_dev, freq_dev, y_dev, X_test, freq_test, y_test


if __name__ == "__main__":
    
    print("Started.")

    # np.random.seed(2)

    word_index, X_train, y_train, freq_train, X_dev, freq_dev, y_dev, X_test, freq_test, y_test = get_data()

    utils.get_custom_objects().update({'custom_activation': Activation(scaled_sigmoid)})

    l2_param = .01

    #prepare inputs
    numeric_input = keras.Input(shape=(3,), name="numeric")
    categorical_input = keras.Input(shape=(114,), name="categorical")
    categorical_features = Dense(32, activation='linear', use_bias=False)(categorical_input)

    frequency_input = keras.Input(shape=(250,), name="most_common_words")

    embedding_matrix, num_tokens, embedding_dim = create_word_embedding(word_index)

    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )

    freq = embedding_layer(frequency_input)
    freq = Flatten()(freq)
    freq = Dense(1024, kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(freq)
    freq = Dense(1024, kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(freq)

    x = layers.concatenate([numeric_input, categorical_features, freq])

    #fully connected netword
    x = BatchNormalization()(x)
    x = Dense(1024, kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(l2_param),  activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dense(512,  kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dense(512,  kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dense(256,  kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(x)
    x = Dense(64,  kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(x)
    x = Dense(32,  kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(x)
    x = Dense(16, kernel_regularizer=regularizers.l2(l2_param), activation='tanh')(x)
    outputs = Dense(1,  kernel_regularizer=regularizers.l2(l2_param), activation='custom_activation')(x)

    optimizer = keras.optimizers.Adam(lr=0.0005)

    model = keras.Model([numeric_input, categorical_input, frequency_input], outputs, name="Script2Score")

    #we use early stopping for regularization
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=300,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    # Compile model
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')

    # Train model
    model.fit(
              {"numeric": X_train.T[:, 114:117], "categorical":  X_train.T[:,:114], "most_common_words": freq_train.T},
              y_train.T,
              batch_size=700,
              epochs=4000,
              callbacks=[early_stopping],
              verbose=1,
              validation_data=({"numeric": X_dev.T[:, 114:117], "categorical":  X_dev.T[:,:114],
                                "most_common_words": freq_dev.T}, y_dev.T)
         )

    score = model.evaluate({"numeric": X_dev.T[:, 114:117], "categorical":  X_dev.T[:,:114],
                            "most common words": freq_dev.T}, y_dev.T)

    print(model.predict({"numeric": X_dev.T[:, 114:117], "categorical":  X_dev.T[:,:114]}))

    # Summary of neural network, saved in 'model' folder
    model.summary()

    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

    model.save('model')
    print(score)