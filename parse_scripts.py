from bs4 import BeautifulSoup
import os



def get_direction_str():
    direction_dict = {}
    files = os.listdir('scripts')
    for file in files:
        print(str(files.index(file)) + ' / ' + str(len(files)))

        html = 'scripts/' + file
        link = open(html)

        try:
            soup = BeautifulSoup(link.read(), features="html.parser")
        except UnicodeDecodeError:
            direction_dict[file] = 'UnicodeDecodeError'
            continue

        for script in soup(["script", "style"]):  # not script in our sense, script in the soup module
            script.extract()

        print("Headers:")
        print(len(soup.find_all('b')))

        direction = map(lambda tag : tag.get_text(), soup.find_all('b')[:400])
        direction_concat = ''.join(direction)
        direction_concat = ''.join(filter(lambda x: x.isalpha() or x == " ", direction_concat)).lower()
        print("len:" + str(len(direction_concat.split()[:150])))
        direction_concat = " ".join(direction_concat.split()[:150])

        direction_dict[file] = direction_concat

        print(direction_concat)
    return direction_concat



