import csv
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import sigmoid
from tensorflow import keras as keras


def scaled_sigmoid(X):
   return 10 * sigmoid(X)


if __name__ == "__main__":
    data_file = "movie_metadata.csv"
    data = []

    with open(data_file, 'r', encoding='iso-8859-1') as f:
        csv_reader = csv.reader(f)
        count = 0
        for row in csv_reader:
            try:
                if "title" in row[0] or row[0] == "":
                    continue
                train_example = row[2:]
                data.append(train_example)
            except IndexError:
                pass

    np.random.seed(1)
    np.random.shuffle(data)

    X = np.array(data)[:, :-2].T.astype(np.float)
    Y = np.array(data)[np.newaxis, :, -1].astype(np.float)

    X_train = X[:, :(math.floor(.8 * X.shape[1]))]
    y_train = Y[:, :(math.floor(.8 * X.shape[1]))]
    X_dev = X[:, (math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    y_dev = Y[:, (math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    X_test = X[:, (math.floor(.9 * X.shape[1])):]
    y_test = Y[:, (math.floor(.9 * X.shape[1])):]

    utils.get_custom_objects().update({'custom_activation': Activation(scaled_sigmoid)})

    model = models.Sequential()
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(256, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='custom_activation'))

    optimizer = keras.optimizers.Adam(lr=0.001)

    # Compile model
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',  # fix
                  metrics=['mean_squared_error'])

    # Train model
    model.fit(X_train.T, y_train.T,
              batch_size=500,
              epochs=1000,
              # callbacks=[plot_losses],
              verbose=1,
              validation_data=(X_dev.T, y_dev.T))

    score = model.evaluate(X_dev.T, y_dev.T)

    print(score)

    # Summary of neural network
    model.summary()

    model.save('model')
