from imdb import IMDb
import os

# create an instance of the IMDb class
ia = IMDb()

'''
# get a movie and print its director(s)
the_matrix = ia.get_movie('0133093')
print(the_matrix)
print(sorted(the_matrix.keys()))
print(the_matrix['box office'])
print(the_matrix['genres'])
print(the_matrix['languages'])
print(the_matrix['language codes'])
print(the_matrix['original air date'])
print(the_matrix['rating'])
print(the_matrix['runtimes'])
print(the_matrix['top 250 rank'])
print(the_matrix['year'])
'''

ids = [imdbid[2: 9] for imdbid in os.listdir('scripts')]
movies = {}
for imdbid in range(len(ids)):
    print(imdbid)
    movies[ids[imdbid]] = {}
    movies[ids[imdbid]]['api object'] = ia.get_movie(ids[imdbid])
print(movies)
