"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
#import scipy as sp
#import operator
import pickle
#import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from time import time

from utils.data_loader import load_movie_titles

# Importing data
#movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
movies_df = pd.read_csv('resources/data/movies.csv')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

train_df = pd.read_csv("resources/data/train.csv")
train_df = train_df.drop("timestamp", axis=1)

title_list = load_movie_titles('resources/data/movies.csv')
list1 = title_list[14930:15200]
list2 = title_list[25055:25255]
list3 = title_list[21100:21200]

movieIds_list1 = movies_df[movies_df['title'].isin(list1)]['movieId'].tolist()
movieIds_list2 = movies_df[movies_df['title'].isin(list2)]['movieId'].tolist()
movieIds_list3 = movies_df[movies_df['title'].isin(list3)]['movieId'].tolist()

selected_movieIds = movieIds_list1 + movieIds_list2 + movieIds_list3

subset_train_df = train_df[train_df['movieId'].isin(selected_movieIds)]

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
#model=pickle.load(open('resources/models/SVD.pkl', 'rb'))
#model=pickle.load(open('resources/models/KNN.pkl', 'rb'))

#ratings = df_subset.pivot_table(index='movieId', columns='userId', values='rating')
#ratings.fillna(0, inplace=True)
#df_train_eng = df_train_eng.drop("timestamp", axis=1)
#print("Ratings:")
#print(ratings.head())

#user_votes = df_subset.groupby('movieId')['rating'].agg('count')
#movie_votes = df_subset.groupby('userId')['rating'].agg('count')

#ratings = ratings.loc[user_votes[user_votes > 10].index,:]
#ratings = ratings.loc[:,movie_votes[movie_votes > 50].index]

#csr_data = csr_matrix(ratings.values)
#ratings.reset_index(inplace=True)

def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """

    # Data preprocessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(df_train,reader)
    a_train = load_df.build_full_trainset()

    predictions = []

    for ui in a_train.all_users():
        
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
        
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """

    # Store the id of users
    id_store=[]

    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)

        predictions.sort(key=lambda x: x.est, reverse=True)

        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)

    # Return a list of user id's
    return id_store

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    #movie_ids = pred_movies(movie_list)

    #df_init_users = df_train[df_train['userId']==movie_ids[0]]
    
    #for userid in movie_ids[1:]:
    #    df_init_users = pd.concat([df_init_users, df_train[df_train['userId']==userid]])

    #df_pivot = df_init_users.pivot_table(index='userId', columns='movieId', values='rating', aggfunc='mean')

    #df_pivot = df_pivot.fillna(0.0)

    #movie_similarity = df_pivot.corr()

    #target_movie_id = movies_df[movies_df['title'].isin(movie_list)]

    #target_movie_id = target_movie_id["movieId"].tolist()

    #if target_movie_id[0] in movie_similarity.columns:
    #    print(target_movie_id[0])
    #    target_movie_similarities1 = movie_similarity[target_movie_id[0]]
    #else:
    #    target_movie_similarities1 = pd.Series([])
        
    #if target_movie_id[1] in movie_similarity.columns:
    #    print(target_movie_id[1])
    #    target_movie_similarities2 = movie_similarity[target_movie_id[1]]
    #else:
    #    target_movie_similarities2 = pd.Series([])
        
    #if target_movie_id[2] in movie_similarity.columns:
    #    print(target_movie_id[2])
    #    target_movie_similarities3 = movie_similarity[target_movie_id[2]]
    #else:
    #    target_movie_similarities3 = pd.Series([])

    #concatenated_series = pd.concat([target_movie_similarities1, target_movie_similarities2, target_movie_similarities3], axis=0)

    #top_indexes = concatenated_series.sort_values(ascending=False).index[:10]

    #recommended_movies = []

    #for i in top_indexes:
    #    look_movieId = movies_df[movies_df['movieId'] == i]
    #    movie_title = look_movieId.iloc[0]['title']
    #    recommended_movies.append(movie_title)

    #time1 = time()

    #print("TIME:", time1)

    ratings = subset_train_df.pivot_table(index='movieId', columns='userId', values='rating')

    #time2 = time()

    #print("Time to pivot:", time2-time1)
    #print("Pivot shape:", ratings.shape)

    ratings.fillna(0, inplace=True)
    csr_data = csr_matrix(ratings.values)
    ratings.reset_index(inplace=True)

    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    model.fit(csr_data)
    
    recommended_movies = []

    rec_1 = get_movie_recommendation(ratings, movie_list[0], model, csr_data)

    rec_2 = get_movie_recommendation(ratings, movie_list[1], model, csr_data)

    rec_3 = get_movie_recommendation(ratings, movie_list[2], model, csr_data)

    all_recommendations = pd.concat([rec_1, rec_2, rec_3], ignore_index=True)

    sorted_recs = all_recommendations.sort_values(by="Distance", ascending=False)

    recs_no_dups = sorted_recs.drop_duplicates(subset='Title', keep='first')

    recs_no_dups = recs_no_dups[~recs_no_dups['Title'].isin(movie_list)]

    recommended_movies = list(recs_no_dups['Title'][:10])

    #rec1 = get_movie_recommendation(ratings, "Iron Man (2008)", model, csr_data)

    #print(rec1)

    return recommended_movies

def get_movie_recommendation(ratings, movie_name, knn, csr_data):
    
    n_movies_to_recommend = 10
    
    #print("Movie name:", movie_name)

    movie_list = movies_df[movies_df['title'].str.contains(movie_name, regex=False)]
    
    #print("Movie list:")
    #print(movie_list)

    if len(movie_list):
        
        movie_idx = movie_list.iloc[0]['movieId']
        
        #print("Movie idx 1:", movie_idx)

        movie_idx = ratings[ratings['movieId'] == movie_idx].index[0]

        #print("Movie idx 2:", movie_idx)
        
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_recommend+1)

        #print("Distances")
        #print(distances)

        #print("Indices")
        #print(indices)
        
        rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        
        #print("Rec Movie Indices:")
        #print(rec_movie_indices)

        recommend_frame = []
        
        for val in rec_movie_indices:
            
            movie_idx = ratings.iloc[val[0]]['movieId']

        #print("Rec Movie Indices:")
            
            idx = movies_df[movies_df['movieId'] == movie_idx].index
            
            recommend_frame.append({'Title':movies_df.iloc[idx]['title'].values[0],'Distance':val[1]})
            
        df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_recommend+1))
        
        return df
    else:
        return "No movies found. Please check your input"