from bs4 import BeautifulSoup
import os
import pickle

# Use global_word_index as a hack, easier to do this in Haskell
global global_word_index
global n_f_i

def get_index(word):
    try:
        return global_word_index[word]
    except KeyError:
        print("miss: " + word)
        return n_f_i


def get_direction_strs(word_index, not_found_index):
    
    try: 
        direction_dict = pickle.load(open("save.p", "rb" ))
        return direction_dict
    except FileNotFoundError:
        print("Must recreate direction dict")

    global global_word_index
    global_word_index = word_index
    global n_f_i
    n_f_i = not_found_index

    direction_dict = {}
    files = os.listdir('scripts')
    for file in files:
        print(str(files.index(file)) + ' / ' + str(len(files)))
        print(file)

        html = 'scripts/' + file
        link = open(html)

        try:
            soup = BeautifulSoup(link.read(), features="html.parser")
        except UnicodeDecodeError:
            direction_dict[file] = 'UnicodeDecodeError'
            continue

        for script in soup(["script", "style"]):  # not script in our sense, script in the soup module
            script.extract()

        # print("Headers:")
        # print(len(soup.find_all('b')))

        direction = map(lambda tag : tag.get_text(), soup.find_all('b')[:400])
        direction_concat = ''.join(direction)
        direction_concat = ''.join(filter(lambda x: x.isalpha() or x == " ", direction_concat)).lower()
        dir_vec = list(map(get_index, direction_concat.split()[:150]))

        dir_vec += [n_f_i] * (150 - len(dir_vec))
        direction_dict[file.split(".")[0]] = dir_vec
    pickle.dump(direction_dict, open("save.p", "wb" ))
    return direction_dict



