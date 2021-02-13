from imdb import IMDb
import os

# create an instance of the IMDb class
ia = IMDb()

# get a movie and print its director(s)
the_matrix = ia.get_movie('0133093')
print(the_matrix)
print(sorted(the_matrix.keys()))
