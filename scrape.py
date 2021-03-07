import os
from urllib.parse import quote
from bs4 import BeautifulSoup
from googlesearch import search
import requests

# Based on the script by nchibana
# https://gist.github.com/nchibana/6cb0d9baee18fc26d32d5824f9647085

BASE_URL = 'http://www.imsdb.com'
SCRIPTS_DIR = 'scripts'

def get_script(relative_link, writer):
    tail = relative_link.split('/')[-1]
    print('fetching %s' % tail)
    script_front_url = BASE_URL + quote(relative_link)
    front_page_response = requests.get(script_front_url)
    front_soup = BeautifulSoup(front_page_response.text, "html.parser")

    try:
        script_link = front_soup.find_all('p', align="center")[0].a['href']
    except IndexError:
        print('%s has no script :(' % tail)
        return None, None
    if script_link.endswith('.html'):
        title = script_link.split('/')[-1].split(' Script')[0]
        results = search(title.split(".html")[0].replace("-", " ").replace("%20", " ") + " imdb " + writer, num_results=10)
        index = 0
        imdb = results[index]
        print(imdb)
        while "/www.imdb.com/title/" not in imdb:
            index += 1
            imdb = results[index]
        assert "/www.imdb.com/title/" in imdb
        imdb = imdb.replace("https://www.imdb.com/title/", "").split("/")[0]
        print(imdb)

        script_url = BASE_URL + script_link
        script_soup = BeautifulSoup(requests.get(script_url).text, "html.parser")
        script_text = script_soup.find_all('td', {'class': "scrtext"})[0]

        if script_text.table is not None:
            script_text.table.decompose()
        if script_text.div is not None:
            script_text.div.decompose()

        # print(script_text.prettify)
        return imdb, script_text.prettify()
    else:
        print('%s is a pdf :(' % tail)
        return None, None


if __name__ == "__main__":
    print("Start")
    response = requests.get('https://imsdb.com/all-scripts.html')
    html = response.text

    soup = BeautifulSoup(html, "html.parser")
    paragraphs = soup.find_all('p')
    if os.path.exists("script_list.txt"):
        os.remove("script_list.txt")
    id_list = open("script_ids.txt", "a")

    for p in paragraphs:
        print(p.i.getText())
        relative_link = p.a['href']
        id, script = get_script(relative_link, p.i.getText())
        if not script:
            continue
        id_list.write(id + "\n")
        with open(os.path.join(SCRIPTS_DIR, id + '.html'), 'w', encoding='utf-8') as outfile:
            outfile.write(script)
