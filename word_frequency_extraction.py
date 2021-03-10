from bs4 import BeautifulSoup
import os
import csv

# BeautifulSoup help found online at
# https://stackoverflow.com/questions/48935514/beautifulsoup-open-local-and-http-html-files


def clean_word(word):
    result = ''
    for ch in word:
        if ch.isalpha() or ch == '\'':
            result += ch
    return result


all_words = []
frequency_dict = {}

files = os.listdir('scripts')
for file in files:
    print(str(files.index(file)) + ' / ' + str(len(files)))
    html = 'scripts\\' + file
    link = open(html)

    try:
        soup = BeautifulSoup(link.read(), features="html.parser")
    except UnicodeDecodeError:
        frequency_dict[file] = 'UnicodeDecodeError'
        continue

    for script in soup(["script", "style"]):  # not script in our sense, script in the soup module
        script.extract()
    text = soup.get_text()
    words = [clean_word(word.lower()) for word in text.split() if clean_word(word.lower()) != '']
    for word in words:
        if word not in all_words:
            all_words.append(word)

    words_dict = {}

    for word in words:
        if word not in words_dict:
            words_dict[word] = 0
        words_dict[word] += 1

    frequency_dict[file] = words_dict

all_words.sort()
print(len(all_words))

with open('word_list.txt', 'w') as f:
    f.write(','.join(all_words))

index_dict = {}

for word in all_words:
    index_dict[word] = all_words.index(word) if word in all_words else -1

top_words_dict = {}

for file in files:
    words_dict = frequency_dict[file]

    if words_dict != 'UnicodeDecodeError':
        top_words = sorted(words_dict, key=lambda x: words_dict[x], reverse=True)
        top_words = [word for word in top_words if word != ''][:1000]
        top_words = top_words
        frequency_vector = [file] + [str(index_dict[word]) for word in top_words if word in index_dict]
    else:
        top_words = [file, 'UnicodeDecodeError']
        frequency_vector = [file, 'UnicodeDecodeError']

    with open('frequency_csv.csv', 'a') as f:
        write = csv.writer(f)
        write.writerow(frequency_vector)

    with open('wop_words_csv.csv', 'a') as f:
        write = csv.writer(f)
        write.writerow([file] + top_words)
