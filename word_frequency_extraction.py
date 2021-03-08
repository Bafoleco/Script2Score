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

files = os.listdir('scripts')
for file in files:
    print(str(files.index(file)) + ' / ' + str(len(files)))
    if files.index(file) == 2:
        break  # Only here to shorten time for testing the program
    html = 'scripts\\' + file
    link = open(html)
    soup = BeautifulSoup(link.read(), features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    words = [clean_word(word.lower()) for word in text.split()]
    for word in words:
        if word not in all_words:
            all_words.append(word)

all_words.sort()
print(len(all_words))

with open('word_list.txt', 'w') as f:
    f.write(','.join(all_words))

index_dict = {}

for word in all_words:
    index_dict[word] = all_words.index(word) if word in all_words else -1

for file in files:
    html = 'scripts\\' + file
    link = open(html)
    soup = BeautifulSoup(link.read(), features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    text = soup.get_text()
    words = [clean_word(word.lower()) for word in text.split()]

    words_dict = {}

    for word in words:
        if word not in words_dict:
            words_dict[word] = 0
        words_dict[word] += 1

    top_words = sorted(words_dict, key=lambda x: words_dict[x], reverse=True)
    top_words = [word for word in top_words if word != ''][:1000]
    top_words = [file] + top_words

    frequency_vector = [index_dict[word] for word in top_words if word in index_dict]
    with open('frequency_csv.csv', 'a') as f:
        write = csv.writer(f)
        write.writerow(top_words)
