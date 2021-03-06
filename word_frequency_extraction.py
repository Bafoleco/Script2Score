import pandas
import html2text
from bs4 import BeautifulSoup
import os
import csv

# BeautifulSoup help found online at
# https://stackoverflow.com/questions/48935514/beautifulsoup-open-local-and-http-html-files

all_words = []

files = os.listdir('scripts')
for file in files:
    print(str(files.index(file)) + ' / ' + str(len(files)))
    if files.index(file) == 10:
        break  # Only here to shorten time for testing the program
    html = 'scripts\\' + file
    link = open(html)
    soup = BeautifulSoup(link.read(), features="html.parser")
    for script in soup(["script", "style"]):
        script.extract()
    text = soup.get_text()
    words = [word.lower() for word in text.split()]
    for word in words:
        if word not in all_words:
            all_words.append(word)

all_words.sort()
print(len(all_words))


html = 'scripts\\tt0133093.html'
link = open(html)
soup = BeautifulSoup(link.read(), features="html.parser")

# kill all script and style elements
for script in soup(["script", "style"]):
    script.extract()    # rip it out

text = soup.get_text()
words = [word.lower() for word in text.split()]

words_dict = {}

for word in words:
    if word not in words_dict:
        words_dict[word] = 0
    words_dict[word] += 1

index_dict = {}

for word in all_words:
    index_dict[word] = all_words.index(word) if word in all_words else -1

#index_vector = [all_words.index(word) if word in all_words else -1 for word in words_dict.keys()]
print('Index Vector: ' + str(index_dict))
print('Word Dictionary: ' + str(words_dict))
