
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import ast
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

url_mov='https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_movies.csv'
url_cred='https://raw.githubusercontent.com/4GeeksAcademy/k-nearest-neighbors-project-tutorial/main/tmdb_5000_credits.csv'

movies = pd.read_csv(url_mov)
credits = pd.read_csv(url_cred)



movies = movies.merge(credits, right_on=['movie_id','title'],left_on=['id','title']).drop('movie_id',axis=1)


movies = movies[['id','title','overview','genres','keywords','cast','crew']]

movies.dropna(inplace=True)
movies.reset_index(inplace=True)


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)


def convert3(obj):
    L = []
    count = 0
    for i in ast.literal_eval(obj):
        if count < 3:
            L.append(i['name'])
        count +=1  
    return L

movies['cast'] = movies['cast'].apply(convert3)



def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L

movies['crew'] = movies['crew'].apply(fetch_director)


movies['overview'] = movies['overview'].apply(lambda x : x.split())



def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)



movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

new_df = movies[['id','title','tags']].copy()

new_df['tags'] = new_df['tags'].apply(lambda x :" ".join(x))



cv = CountVectorizer(max_features=5000 ,stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()


similarity = cosine_similarity(vectors)


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0] ##fetching the movie index
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate( distances)),reverse =True , key = lambda x:x[1])[1:6]
    
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

