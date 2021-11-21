# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
""" 
from flask import Flask , redirect, url_for, render_template , request
import numpy as np
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity 


#creating a function that creates a similarity matrix if it doesn't exisit 
def create_similarity():
    data = pd.read_csv(r"C:\Users\pgp19\desk\RCMD\preprocessed_data.csv")
    # creating combination list 
    # making a new column containing combination of all the features our recommendation system will be based on 
    data["combination"] = data["plot"]+" "+data["genre"]+" "+data["Directed by"]+" "+data["Produced by"]+" "+data["Starring"]+" "+data["Music by"]+" "+data["Distributed by"]
    # creating a count matrix
    #cv = CountVectorizer()
    #count_matrix = cv.fit_transform(data["combination"])
    # creating a similarity score matrix 
    tvf = TfidfVectorizer(min_df = 3, max_features = None, 
                         strip_accents = "unicode", analyzer = "word",
                         token_pattern=r'\w{1,}',ngram_range=(1, 3),
                         stop_words = 'english')
    
    tvf_matrix = tvf.fit_transform(data["combination"])

    similarity = cosine_similarity(tvf_matrix)
    return data , similarity 


# defining a function that recommends top 5 most similar movies 
def recommend_movie(m): 
    data, similarity = create_similarity()
        # check if movie is in our database or not 
    if m not in data["title"].unique():
        return("This movie is not in out database. \n Please check if you spelled it correct. Try capitalizing the first letter!")
    else:
        #getting the index of the movie in the dataframe 
        i = data.loc[data["title"]==m].index[0]
            
            ##fetching the row containing the similarity score of the movie from the similarity matrix and enumerate it 
        lst = list(enumerate(similarity[i]))
            
            ##sorting this list in descending order based on similarity score 
        lst = sorted(lst,key = lambda x:x[1], reverse = True)
            
            ## taking the top 5 movies, ignoring the first one since its the same movie 
        lst = lst[1:6]
            
            #making an empty list that will contain all 5 recommended movies 
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data["title"][a])
        return l
        

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    r = recommend_movie(movie)
    movie = movie.upper()
    if type(r)==type('string'):
        return render_template('recommend.html',movie=movie,r=r,t='s')
    else:
        return render_template('recommend.html',movie=movie,r=r,t='l')
    
if __name__ == '__main__':
    app.run()
