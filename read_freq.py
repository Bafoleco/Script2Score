import csv

frequency_file = "frequency_csv.csv"
freq_vecs = {}
bad_ids = []

with open(frequency_file, 'r') as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        if len(row) > 260:
            id = row[0][: row[0].find('.')]
            if row[1] == 'UnicodeDecodeError':
                bad_ids.append(id)
                continue
            freq_vecs[id] = [int(idx) for idx in row[1: 251]]

word_idx_file = "word_list.txt"

with open(word_idx_file, 'r') as f:
    text = f.read()
    words = text.split()
