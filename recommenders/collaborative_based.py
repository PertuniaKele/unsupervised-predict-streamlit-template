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
import scipy as sp
import operator
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
#from sklearn.neighbors import NearestNeighbors
from time import time

# Importing data
#movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
movies_df = pd.read_csv('resources/data/movies.csv')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)
#df_subset = pd.read_csv('resources/data/movies_subset.csv')
#df_train = pd.read_csv("resources/data/train.csv")
#df_train = df_train.drop("timestamp", axis=1)
#df_subset = df_train[:10]

#print("Subset:")
#print(df_subset.head())

#model = SVD(n_factors=20, n_epochs=10)
#model.fit(ratings_df)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))
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
    load_df = Dataset.load_from_df(ratings_df,reader)
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
        print("i in Movie List:", i)
        predictions = prediction_item(item_id = i)

        #print(predictions[:100])

        predictions.sort(key=lambda x: x.est, reverse=True)

        #print(predictions[:10])

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

    indices = pd.Series(movies_df['title'])
    movie_ids = pred_movies(movie_list)

    #print(movie_ids)

    df_init_users = ratings_df[ratings_df['userId']==movie_ids[0]]
    
    for userid in movie_ids[1:]:
        df_init_users = pd.concat([df_init_users, ratings_df[ratings_df['userId']==userid]])

    #cosine_sim = cosine_similarity(np.array(df_init_users), np.array(df_init_users))

    #df_init_users = df_init_users.reset_index(drop=True)

    df_pivot = df_init_users.pivot_table(index='userId', columns='movieId', values='rating', aggfunc='mean')

    #cosine_sim = cosine_similarity(df_init_users.pivot(index='userId', columns='movieId', values='rating'), 
    #                           df_init_users.pivot(index='userId', columns='movieId', values='rating'))

    df_pivot = df_pivot.fillna(0.0)

    #print("Pivot Shape:", df_pivot.shape)

    cosine_sim = cosine_similarity(df_pivot, df_pivot)

    selected_movie_indices = movies_df[movies_df['title'].isin(movie_list)].index

    #print("Selected Movie Indices")
    #print(selected_movie_indices)

    #print(cosine_sim.shape)
    #print(cosine_sim)

    movie_mapping = {df_pivot.columns.get_loc(movie_id): movie_index for movie_index, movie_id in enumerate(df_pivot.columns)}

    #print("Movie Map")
    #print(movie_mapping)

    #print(df_init_users.head(50))
    #print(df_init_users.shape)

    #print("User Count:", df_init_users['userId'].unique())

    #util_matrix = df_init_users.pivot_table(index=['userId'], columns=['movieId'], values='rating')

    #util_matrix_norm = util_matrix.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
    #util_matrix_norm.fillna(0, inplace=True)
    #util_matrix_norm = util_matrix_norm.T
    #util_matrix_norm = util_matrix_norm.loc[:, (util_matrix_norm != 0).any(axis=0)]
    #util_matrix_sparse = sp.sparse.csr_matrix(util_matrix_norm.values)

    #user_similarity = cosine_similarity(util_matrix_sparse.T)

    #user_sim_df = pd.DataFrame(user_similarity, index = util_matrix_norm.columns, columns = util_matrix_norm.columns)

    #recommended_ids = get_recommendations(user_sim_df, util_matrix_norm)

    recommended_movies = []

    #for movie_index in selected_movie_indices:
    #    similarity_scores = cosine_sim[movie_index]
    #    similar_movie_indices = np.argsort(similarity_scores)[-top_n-1:-1]  # Exclude the movie itself
    #
    #    top_movies = movies_df.iloc[similar_movie_indices]['title'].tolist()
    #    recommended_movies.extend(top_movies)


    for movie_index in selected_movie_indices:
        df_pivot_index = movie_mapping.get(movie_index)
        if df_pivot_index is not None:
            similarity_scores = cosine_sim[df_pivot_index]
            similar_movie_indices = np.argsort(similarity_scores)[-top_n-1:-1]  # Exclude the movie itself

            top_movies = movies_df.iloc[similar_movie_indices]['title'].tolist()
            recommended_movies.extend(top_movies)

    #for id in recommended_ids:
    #    movie_title = movies_df[movies_df["movieId"]==id]["title"].iloc[0]
    #    recommended_movies.append(movie_title)

    #print(recommended_ids)
    #print(recommended_movies)
        
    # Choose top 50
    #top_50_indexes = list(listings.iloc[1:50].index)
    #print("Top 50 Indexes:", top_50_indexes) 
    
    # Removing chosen movies
    #top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    #print("Filtered Top Indexes:", top_indexes)
    
    #for i in top_50_indexes[:top_n]:
    #    recommended_movies.append(list(movies_df['title'])[i])

    # Get the indices of the top-n predicted ratings
    #top_indices = sorted(range(len(predictions)), key=lambda i: predictions[i], reverse=True)[:top_n]

    # Get the corresponding movie titles
    #recommended_movies = indices[top_indices].tolist()

    #rec_1 = pd.Series(get_movie_recommendation(movie_list[0]))
    #rec_2 = pd.Series(get_movie_recommendation(movie_list[1]))
    #rec_3 = pd.Series(get_movie_recommendation(movie_list[2]))
    
    #all_recommendations = pd.concat([rec_1, rec_2, rec_3], ignore_index=True)

    #sorted_recs = all_recommendations.sort_values(ascending=False)

    #top_10 = sorted_recs[:10]

    #print(top_10)

    #recommended_movies = top_10['title'].tolist()

    return recommended_movies

#def get_recommendations(final_df, norm_df):
    
#    sim_users = final_df.sort_values(by="userId", ascending=False).index[1:10]
#    print(sim_users)
#    favorite_user_items = [] # <-- List of highest rated items gathered from the k users  
#    most_common_favorites = {} # <-- Dictionary of highest rated items in common for the k users
    
#    for i in sim_users:
        # Maximum rating given by the current user to an item 
#        max_score = norm_df.loc[:, i].max()
        # Save the names of items maximally rated by the current user   
#        favorite_user_items.append(norm_df[norm_df.loc[:, i]==max_score].index.tolist())
        
    # Loop over each user's favorite items and tally which ones are 
    # most popular overall.
#    for item_collection in range(len(favorite_user_items)):
#        for item in favorite_user_items[item_collection]: 
#            if item in most_common_favorites:
#                most_common_favorites[item] += 1
#            else:
#                most_common_favorites[item] = 1
    
    # Sort the overall most popular items and return the top-N instances
#    sorted_list = sorted(most_common_favorites.items(), key=operator.itemgetter(1), reverse=True)[:10]
#    top_N = [x[0] for x in sorted_list]

#    return top_N