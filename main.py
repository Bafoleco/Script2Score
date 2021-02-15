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