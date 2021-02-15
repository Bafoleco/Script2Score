import csv
import math
import numpy as np
#import tensorflow as tf

print("Hello, world!")

data_file = "movie_metadata.csv"
X = []
Y = []

with open(data_file, 'r') as f:
    csv_reader = csv.reader(f)
    count = 0
    for row in csv_reader:
        try:
            if "title" in row[0]:
                continue
            train_example = row[2:-2]
            X.append(train_example)
            train_output = row[-1]
            Y.append(train_output)
        except IndexError:
            pass

X = np.array(X).T.astype(np.float)
Y = np.array(Y, ndmin=2).astype(np.float)

#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

"""
X_train = 
y_train = 
X_test = 
y_test = 

model = models.Sequential()
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(10, activation='relu'))


# Compile model
model.compile(optimizer='adam',
              loss='categorical_crossentropy', #fix
              metrics=['accuracy'])

# Train model
model.fit(X_train, y_train,
          #batch_size=BATCH_SIZE,
         # epochs=EPOCHS,
          #callbacks=[plot_losses],
         # verbose=1,
          validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)

# Summary of neural network
model.summary()
"""