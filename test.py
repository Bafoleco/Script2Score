from utils import predict_score
import csv
import math
import tensorflow.keras as keras
import numpy as np

# print(predict_score("83987"))

data_file = "movie_metadata.csv"
model = keras.models.load_model('model')

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


print(X_test.T.shape)
print(X_train.T.shape)

print(model.evaluate({"numeric": X_dev.T[:, 111:115], "categorical":  X_dev.T[:,:111]}, y_dev.T))
print(model.evaluate({"numeric": X_test.T[:, 111:115], "categorical":  X_test.T[:,:111]}, y_test.T))

print("predicted score: " + predict_score())