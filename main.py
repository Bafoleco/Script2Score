import csv
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras import utils
from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import sigmoid
from tensorflow import keras as keras
from tensorflow.keras import regularizers


#custom activation to be used
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

    X = np.array(data)[:,:-2].T.astype(np.float)
    Y = np.array(data)[np.newaxis,:,-1].astype(np.float)


    #seperation of dataset into train and test sets
    X_train = X[:,:(math.floor(.8 * X.shape[1]))]
    y_train = Y[:,:(math.floor(.8 * X.shape[1]))]
    X_dev = X[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    y_dev = Y[:,(math.floor(.8 * X.shape[1])):(math.floor(.9 * X.shape[1]))]
    X_test = X[:,(math.floor(.9 * X.shape[1])):]
    y_test = Y[:,(math.floor(.9 * X.shape[1])):]

    np.random.seed(2)

    utils.get_custom_objects().update({'custom_activation': Activation(scaled_sigmoid)})


    #fully connected neural network
    model = models.Sequential()
    model.add(BatchNormalization())
    model.add(Dense(512, kernel_regularizer=regularizers.l2(1e-4),  activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(512,  kernel_regularizer=regularizers.l2(1e-4), activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(512,  kernel_regularizer=regularizers.l2(1e-4), activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dense(256,  kernel_regularizer=regularizers.l2(1e-4), activation='tanh'))
    model.add(Dense(64,  kernel_regularizer=regularizers.l2(1e-4), activation='tanh'))
    model.add(Dense(16,  kernel_regularizer=regularizers.l2(1e-4), activation='tanh'))
    model.add(Dense(1,  kernel_regularizer=regularizers.l2(1e-4), activation='custom_activation'))

    optimizer = keras.optimizers.Adam(lr=0.0006)


    #we use early stopping for regularization
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=500,
        verbose=0,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    )

    # Compile model
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error', #fix
                  metrics=['mean_squared_error'])

    # Train model
    model.fit(X_train.T, y_train.T,
              batch_size=600,
              epochs=1500,
              callbacks=[early_stopping],
              verbose=1,
              validation_data=(X_dev.T, y_dev.T))


    score = model.evaluate(X_dev.T, y_dev.T)

    print(model.predict(X_dev.T))

    # Summary of neural network, saved in 'model' folder
    model.summary()

    model.save('model')
