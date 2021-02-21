import csv
import math
import numpy as np
from tensorflow import keras

data_file = "movie_metadata.csv"
model = keras.models.load_model('model')
id_map = dict()

with open(data_file, 'r', encoding='iso-8859-1') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        try:
            if "title" in row[0] or row[0] == "":
                continue
            id_map[row[1]] = row[2:-2]
        except IndexError:
            pass


def predict_score(IMDB_ID):
    data = np.array(id_map[IMDB_ID]).reshape(115, 1).astype(np.float).T
    return model.predict(data)