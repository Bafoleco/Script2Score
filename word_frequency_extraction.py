import pandas
import html2text
from bs4 import BeautifulSoup

# BeautifulSoup help found online at
# https://stackoverflow.com/questions/48935514/beautifulsoup-open-local-and-http-html-files

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

print(words_dict)
