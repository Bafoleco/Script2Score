from bs4 import BeautifulSoup
import os


meta_map = {}

files = os.listdir('scripts')
for file in files:
    print(str(files.index(file)) + ' / ' + str(len(files)))

    html = 'scripts/' + file
    link = open(html)

    try:
        soup = BeautifulSoup(link.read(), features="html.parser")
    except UnicodeDecodeError:
        frequency_dict[file] = 'UnicodeDecodeError'
        continue

    for script in soup(["script", "style"]):  # not script in our sense, script in the soup module
        script.extract()

    print("Headers:")
    print(len(soup.find_all('b')))

    meta_script = soup.find_all('b')[:50]

    start_string 





