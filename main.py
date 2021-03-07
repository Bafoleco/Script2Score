import csv
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
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
def create_word_embedding():
    path_to_glove_file = os.path.join(
        os.path.expanduser("~"), ".keras/datasets/glove.6B.100d.txt"
    )

    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    print("Found %s word vectors." % len(embeddings_index))

    num_tokens = len(voc) + 2
    embedding_dim = 100
    hits = 0
    misses = 0

    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    print("Converted %d words (%d misses)" % (hits, misses))


def get_data():
    data_file = "movie_metadata.csv"
    data = []

    with open(data_file, 'r', encoding='iso-8859-1') as f:
        csv_reader = csv.reader(f)
        count = 0
        for row in csv_reader:
            try:

                if "title" in row[0] or row[0] == "":
                    print(row[113])
                    continue
                train_example = row[2:]
                data.append(train_example)
            except IndexError:
                pass

    np.random.seed(1)
    np.random.shuffle(data)

    X = np.array(data)[:,:-2].T.astype(np.float)
    Y = np.array(data)[np.newaxis,:,-1].astype(np.float)

    #seperation of dataset into train and test sets
    X_train = X[:,:(math.floor(.8 * X.shape[1]))]
    y_train = Y[:,:(math.floor(.8 * X.shape[1]))]
    X_dev = X[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    y_dev = Y[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    X_test = X[:,(math.floor(.9 * X.shape[1])):]
    y_test = Y[:,(math.floor(.9 * X.shape[1])):]
    return X_train, y_train, X_dev, y_dev, X_test, y_test;

if __name__ == "__main__":

    # np.random.seed(2)

    X_train, y_train, X_dev, y_dev, X_test, y_test = get_data()



    utils.get_custom_objects().update({'custom_activation': Activation(scaled_sigmoid)})


    #fully connected neural network
    numeric_input = keras.Input(shape=(4,), name="numeric")
    categorical_input = keras.Input(shape=(111,), name="categorical")

    categorical_features = Dense(100, activation='linear', use_bias=False)(categorical_input)
    x = layers.concatenate([numeric_input, categorical_features])

    """
    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,
    )    
    """

    x = BatchNormalization()(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(1e-3),  activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dense(512,  kernel_regularizer=regularizers.l2(1e-3), activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dense(512,  kernel_regularizer=regularizers.l2(1e-3), activation='tanh')(x)
    x = BatchNormalization()(x)
    x = Dense(256,  kernel_regularizer=regularizers.l2(1e-3), activation='tanh')(x)
    x = Dense(64,  kernel_regularizer=regularizers.l2(1e-3), activation='tanh')(x)
    x = Dense(16,  kernel_regularizer=regularizers.l2(1e-3), activation='tanh')(x)
    outputs = Dense(1,  kernel_regularizer=regularizers.l2(1e-3), activation='custom_activation')(x)

    optimizer = keras.optimizers.Adam(lr=0.0004)

    model = keras.Model([numeric_input, categorical_input], outputs, name="Script2Score")

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

    print(X_train.T.shape)
    print(X_train.T[:, 111:115].shape)
    print(y_train.T.shape)

    # Train model
    model.fit(
              {"numeric": X_train.T[:, 111:115], "categorical":  X_train.T[:,:111]},
              y_train.T,
              batch_size=700,
              epochs=2000,
              callbacks=[early_stopping],
              verbose=1,
              validation_data=({"numeric": X_dev.T[:, 111:115], "categorical":  X_dev.T[:,:111]}, y_dev.T)
         )


    score = model.evaluate({"numeric": X_dev.T[:, 111:115], "categorical":  X_dev.T[:,:111]}, y_dev.T)

    print(model.predict({"numeric": X_dev.T[:, 111:115], "categorical":  X_dev.T[:,:111]}))

    # Summary of neural network, saved in 'model' folder
    model.summary()

    keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

    model.save('model')
    print(score)

